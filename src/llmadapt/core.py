import os
import json
import inspect
import urllib.request
import urllib.error
import threading
import asyncio
import time
from typing import get_type_hints, Callable

from .compressor import CompressionPolicy


class Conversation:
    """ Conversation Class:
    This needs to manage chat memory and has to have information of where the information came from, 
    such as from a tool, model or the user.
    This then needs to be able to send such memory when asked for based on the syntax and structure 
    required depending on the AI model used.
    Key Providers: Gemini, OpenAI, Anthropic, Ollama (for local models).
    These providers should naturally work within the library.
    Custom Providers are possible to use but a function to specify the structure is required. """
    
    def __init__(self, system_instruction: str = ""):
        self.history = []
        self.system_instruction = system_instruction
        self.model_role = "assistant"

        if self.system_instruction:
            self.history.append({"role": "system", "content": self.system_instruction})

    def add_user_msg(self, text: str):
        self.history.append({"role": "user", "content": text})
    
    def add_model_msg(self, text: str = None, tool_calls: list = None, role: str = None,
                       native: dict = None, native_provider: str = None):
        if role is None:
            role = self.model_role

        msg = {"role": role} 
        if text:
            msg["content"] = text
        if tool_calls:
            normalized_calls = []
            for tc in tool_calls:
                func = tc.get("function") or {}
                args = func.get("arguments") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        pass
                normalized_calls.append({
                    "id": tc.get("id"),
                    "type": "function",
                    "function": {
                        "name": func.get("name"),
                        "arguments": args
                    }
                })
            msg["tool_calls"] = normalized_calls

        if native is not None:
            msg["_native"] = native
            msg["_native_provider"] = native_provider
        self.history.append(msg)
    
    def add_tool_response(self, function_name: str, output: str, tool_call_id: str = None):
        msg = {
            "role": "tool",
            "name": function_name,
            "content": output
        }
        if tool_call_id:
            msg["tool_call_id"] = tool_call_id
        self.history.append(msg)

    def export_for(self, provider: str, special_format: Callable[[list], list] = None) -> list:
        """Translates local flat memory frames into target external API layouts."""
        if special_format and callable(special_format):
            return special_format(self.history)

        if provider in ["ollama", "openai", "custom"]:
            cleaned_history = []
            for msg in self.history:
                cleaned_msg = {k: v for k, v in msg.items() if not k.startswith("_")}
                
                if provider in ["openai", "custom"] and "tool_calls" in cleaned_msg:
                    stringified_calls = []
                    for tc in cleaned_msg["tool_calls"]:
                        func = tc.get("function") or {}
                        args = func.get("arguments") or {}
                        if not isinstance(args, str):
                            args = json.dumps(args)
                        
                        stringified_calls.append({
                            "id": tc.get("id"),
                            "type": "function",
                            "function": {
                                "name": func.get("name"),
                                "arguments": args
                            }
                        })
                    cleaned_msg["tool_calls"] = stringified_calls
                
                cleaned_history.append(cleaned_msg)
            return cleaned_history
            
        elif provider == "anthropic":
            return self._export_anthropic()
            
        elif provider == "gemini":    
            return self._export_gemini()

    def change_system_instruction(self, new_instruction: str):
        self.system_instruction = new_instruction
        updated = False
        for msg in self.history:
            if msg["role"] == "system":
                msg["content"] = self.system_instruction
                updated = True
                break
        if not updated and self.system_instruction:
            self.history.insert(0, {"role": "system", "content": self.system_instruction})

    def change_model_role(self, new_role: str):
        self.model_role = new_role

    def _export_anthropic(self) -> list:
        """Structures conversation blocks to comply with Anthropic context constraints."""
        out = []
        pending_tool_results = []
        emitted_native_ids = set() 

        def flush():
            if pending_tool_results:
                out.append({"role": "user", "content": pending_tool_results.copy()})
                pending_tool_results.clear()

        for msg in self.history:
            if msg["role"] == "system":
                continue

            if msg.get("_native_provider") == "anthropic" and "_native" in msg:
                native_hash = json.dumps(msg["_native"], sort_keys=True)
                if native_hash in emitted_native_ids:
                    continue
                emitted_native_ids.add(native_hash)
                flush()
                out.append(msg["_native"])
                continue

            if msg["role"] == "tool":
                pending_tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id", ""),
                    "content": msg["content"]
                })
                continue

            flush()

            if msg["role"] == "user":
                out.append({"role": "user", "content": msg["content"]})
            elif msg["role"] == "assistant":
                content = []
                if msg.get("content"):
                    content.append({"type": "text", "text": msg["content"]})
                for tool_call in msg.get("tool_calls", []):
                    content.append({
                        "type": "tool_use",
                        "id": tool_call["id"],
                        "name": tool_call["function"]["name"],
                        "input": tool_call["function"]["arguments"]
                    })
                out.append({"role": "assistant", "content": content if content else (msg.get("content") or "")})

        flush()
        return out

    def _export_gemini(self) -> list:
        """Structures conversation blocks to comply with Gemini context constraints."""
        gemini_history = []
        for msg in self.history:
            if msg["role"] == "system": 
                continue 
            if msg.get("_native_provider") == "gemini" and "_native" in msg:
                gemini_history.append(msg["_native"])
                continue
            if msg["role"] == "user":
                gemini_history.append({"role": "user", "parts": [{"text": msg["content"]}]})
            elif msg["role"] == self.model_role:
                parts = []
                if msg.get("content"):
                    parts.append({"text": msg["content"]})
                if "tool_calls" in msg:
                    for tool_call in msg["tool_calls"]:
                        function_call_part = {
                            "name": tool_call["function"]["name"],
                            "args": tool_call["function"]["arguments"]
                        }
                        if tool_call.get("id"):
                            function_call_part["id"] = tool_call["id"]
                        parts.append({"functionCall": function_call_part})
                gemini_history.append({"role": "model", "parts": parts})
            elif msg["role"] == "tool":
                function_response_part = {"name": msg["name"], "response": {"result": msg["content"]}}
                if msg.get("tool_call_id"):
                    function_response_part["id"] = msg["tool_call_id"]
                gemini_history.append({
                    "role": "user",
                    "parts": [{"functionResponse": function_response_part}]
                })
        return gemini_history

class ToolRegistry:
    """ Tool Registry Class:
    This class needs to handle tool storage and usage.
    Methods to register tool use tool.
    It should be able to export for all AI models and work with all providers.
    Again, custom providers are possible to use but a function to specify the structure is required."""

    def __init__(self):
        self.functions_maps = {}
        self.schemas = {} 
    
    def register(self, python_function, schema: dict = None):
        if schema is None:
            func_name = python_function.__name__
            func_doc = inspect.getdoc(python_function) or "No description available."
            type_mapping = {
                str: "string", int: "integer", float: "number", bool: "boolean"
            }

            type_hints = get_type_hints(python_function)
            signature = inspect.signature(python_function)
            properties = {}
            required_params = []

            for param_name, param in signature.parameters.items():
                param_type = type_hints.get(param_name, str)
                gemini_type = type_mapping.get(param_type, "string")

                properties[param_name] = {
                    "type": gemini_type,
                    "description": f"The {param_name} parameter"
                }
                if param.default == inspect.Parameter.empty:
                    required_params.append(param_name)
            
            schema = {
                "name": func_name,
                "description": func_doc,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required_params
                }
            }
        else:
            func_name = schema["name"]

        self.functions_maps[func_name] = python_function
        self.schemas[func_name] = schema 
    
    def execute(self, name: str, args: dict) -> str:
        """Invokes a registered tool, safely converting crashes into explicit textual logs."""
        if name not in self.functions_maps:
            return f"Error: Tool '{name}' is not registered in this system."
            
        try:
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    # Auto-wrap single primitive string inputs into standard key payload
                    sig = inspect.signature(self.functions_maps[name])
                    params = list(sig.parameters.keys())
                    if len(params) == 1:
                        args = {params[0]: args}

            if not isinstance(args, dict):
                return f"Error: Arguments for tool '{name}' must be passed as a dictionary."

            res = self.functions_maps[name](**args)
            return self._stringify_tool_output(res)
        except Exception as e:
            error_type = e.__class__.__name__
            return f"Tool Execution Failure ({error_type}): {str(e)}. Please adjust your input arguments."

    def export_for(self, provider: str, special_format: Callable[[list], list] = None) -> list:
        if not self.schemas:
            return []

        schema_list = list(self.schemas.values()) 

        if special_format and callable(special_format):
            return special_format(schema_list)

        if provider in ["ollama", "openai"]:
            return [{"type": "function", "function": schema} for schema in schema_list]
        elif provider == "anthropic":
            return [{"name": s["name"], "description": s["description"], "input_schema": s["parameters"]} for s in schema_list]
        elif provider == "gemini":
            return [{"functionDeclarations": schema_list}]

    @staticmethod
    def _stringify_tool_output(out) -> str:
        """Serializes tool return values into valid JSON strings for model consumption."""
        if isinstance(out, str):
            return out
        try:
            return json.dumps(out)
        except TypeError:
            return str(out)

class Agent:
    """ Agent Class:
    This is the core agent class that needs to have several key features:
    This agent handles the Conversation and Tool object.
    When an AI responds asking for the results of certain tools, this must run it; however, there 
    should be a max to how many times this iteration can happen - function to change max iterations.
    A function to switch APIs and all - will make later additions easier.
    Add tool and send request to AI model functions are needed.
    An overall Chat function that handles sending the payload to the AI model and handles the response. """
    
    DEFAULT_MODELS = {
        "gemini": "gemini-3.5-flash",
        "anthropic": "claude-3-5-sonnet-20241022",
        "openai": "gpt-4o-mini",
        "ollama": "llama3.1:8b"
    }

    def __init__(self, provider: str, model: str = None, base_url: str = None,
                 api_key: str = None, is_local: bool = False, system_instruction: str = "", max_tool_iterations: int = 6,
                 max_tokens: int = None, compression_policy: CompressionPolicy = None):
        self.conversation = Conversation(system_instruction=system_instruction)
        self.tool_registry = ToolRegistry()
        self.change_api(provider=provider, model=model, base_url=base_url, api_key=api_key)
        self.set_max_tool_iterations(max_tool_iterations)
        self.max_tokens = max_tokens
        self.thinking_stage = "initial state"

        self.is_local = is_local

        # Defaults to a disabled policy (compress() is a no-op passthrough) so any existing
        # Agent(...) call that doesn't pass this keeps behaving exactly like it did before -
        # see CompressionPolicy in compressor.py for why this is a policy object rather than
        # the agent owning/constructing a compressor instance.
        self.compression_policy = compression_policy or CompressionPolicy()

        # Real per-provider usage, captured from each response as it comes back
        # (see _extract_usage/_record_usage) - not estimated. company.py's budget
        # governance (budget.py) reads total_tokens_used directly; usage_log keeps
        # the per-call detail in case anything needs to reconstruct it later.
        self.total_tokens_used = 0
        self.usage_log = []

    def reset_usage(self):
        """Zeroes the usage counters. Useful for a long-lived Agent being
        reused across many separate tasks where each task's cost should be
        measured independently."""
        self.total_tokens_used = 0
        self.usage_log = []

    def _extract_usage(self, res: dict) -> dict:
        """Pulls real input/output token counts out of a raw provider
        response. Every provider shapes this differently and some responses
        (e.g. a scripted test double) won't have it at all - missing fields
        default to 0 rather than raising, since usage reporting is a
        best-effort governance signal, not something that should ever break
        an actual chat turn."""
        try:
            if self.provider == "anthropic":
                u = res.get("usage") or {}
                return {"input": u.get("input_tokens", 0) or 0, "output": u.get("output_tokens", 0) or 0}
            if self.provider in ("openai", "custom"):
                u = res.get("usage") or {}
                return {"input": u.get("prompt_tokens", 0) or 0, "output": u.get("completion_tokens", 0) or 0}
            if self.provider == "gemini":
                u = res.get("usageMetadata") or {}
                return {"input": u.get("promptTokenCount", 0) or 0, "output": u.get("candidatesTokenCount", 0) or 0}
            if self.provider == "ollama":
                return {"input": res.get("prompt_eval_count", 0) or 0, "output": res.get("eval_count", 0) or 0}
        except Exception:
            pass
        return {"input": 0, "output": 0}

    def _record_usage(self, res: dict) -> None:
        usage = self._extract_usage(res)
        turn_total = usage["input"] + usage["output"]
        self.total_tokens_used += turn_total
        self.usage_log.append({"input": usage["input"], "output": usage["output"], "total": turn_total})

    def tool(self, python_function: Callable, schema: dict = None):
        """Decorator to register a Python function as a tool with optional schema."""
        if python_function is None:
            def decorator(func):
                self.add_tool(func, schema)
                return func
            return decorator
        self.add_tool(python_function, schema)
        return python_function
    
    def _update_stage(self, stage: str, detail: str = "", thinking_visible: bool = True) -> None:
        """Internal helper to safely mutate system tracking state and print visible console pipelines."""
        self.thinking_stage = stage
        if not thinking_visible: return
        if detail:
            print(f"[{self.provider.upper()}] ➔ {stage}: {detail}", end="\n", flush=True)
        else:
            print(f"[{self.provider.upper()}] ➔ {stage}...", end="\n", flush=True)

    def set_max_tool_iterations(self, n: int):
        """Sets the upper threshold for consecutive tool execution cycles."""
        if not isinstance(n, int) or n < 1:
            raise ValueError("max_tool_iterations must be an integer >= 1")
        self.max_tool_iterations = n

    def change_api(self, provider: str, model: str = None, base_url: str = None, api_key: str = None):
        """Re-routes active transport endpoints with fluent default model fallbacks."""
        self.provider = provider.lower()
        self.model = model or self.DEFAULT_MODELS.get(self.provider, "custom-model")
        self.api_key = api_key

        if base_url:
            self.url = base_url
        elif self.provider == "gemini":
            self.url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        elif self.provider == "openai":
            self.url = "https://api.openai.com/v1/chat/completions"
        elif self.provider == "anthropic":
            self.url = "https://api.anthropic.com/v1/messages"
        elif self.provider == "ollama":
            self.url = "http://localhost:11434/api/chat"
        else:
            self.url = ""

        if not self.api_key and self.provider != "ollama":
            self.api_key = os.getenv(f"{self.provider.upper()}_API_KEY")

    def add_tool(self, python_function: Callable, schema: dict = None):
        self.tool_registry.register(python_function, schema)

    def _send_request(self, payload: dict, headers: dict) -> dict:
        """Executes native standard library post requests using zero external code."""
        if not self.url:
            raise ValueError(f"API endpoint URL is not configured for provider '{self.provider}'. "
                             f"Please supply a valid base_url when calling change_api().")

        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self.url, data=data_bytes, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"\n[{self.provider.upper()} HTTP Error {e.code}]: {e.read().decode('utf-8')}")
        except urllib.error.URLError as e:
            raise RuntimeError(f"\n[Network Unreachable]: Check route {self.url}. Reason: {e.reason}")

    def _execute_with_thread(self, func: Callable, error_container: list, results_container: list, *args, **kwargs):
        """Executes any worker function in a separate thread and safely collects its outputs or crashes."""
        
        def worker():
            """Internal closure block that catches target output signatures."""
            try:
                # Execute the actual target function with passed arguments
                res = func(*args, **kwargs)
                results_container.append(res)
            except Exception as e:
                # Capture target code crashes (like HTTP errors) so the main thread knows about them
                error_container.append(e)

        thread = threading.Thread(target=worker)
        thread.daemon = True
        thread.start()
        
        return thread
            

    def chat(self, user_input: str, custom_format_func: Callable[[list], list] = None, max_tokens: int = None,
             thinking_visible: bool = True, spinner: bool = True) -> str:
        self.conversation.add_user_msg(user_input)
        
        return self.generate(custom_format_func, max_tokens=max_tokens, thinking_visible=thinking_visible)

    def generate(self, custom_format_func: Callable[[list], list] = None, max_tokens: int = None, 
                 thinking_visible: bool = True, spinner: bool = True) -> str:
        """Generate response based on the current conversation and tools."""
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        self._update_stage("initial state", thinking_visible=thinking_visible)
        
        iterations = 0 
        while True:
            iterations += 1
            if iterations > self.max_tool_iterations:
                raise RuntimeError(f"Exceeded max_tool_iterations ({self.max_tool_iterations})")

            history = self.conversation.export_for(self.provider, special_format=custom_format_func)
            tools = self.tool_registry.export_for(self.provider, special_format=custom_format_func)
            headers = {"Content-Type": "application/json"}

            # --- 1. COMPILE THE TARGET PAYLOAD ---
            payload = self.gen_payload(history, tools, headers, max_tokens)

            # --- 2. SEND THE REQUEST ---
            self._update_stage("Request sent to AI", f"Payload Size: {len(history)} turns", thinking_visible=thinking_visible)
            errors, results = [], []            
            thread = self._execute_with_thread(self._send_request, errors, results, payload, headers)
            self.spinner(spinner, thinking_visible, thread)

            # Check if the background thread threw a network exception
            if errors: raise errors[0]
            # Unpack the completed parsed json data dictionary
            res = results[0]

            # --- 3. PROCESS COMPLETED RESPONSE OBJECTS ---
            output = self.process_response(res, thinking_visible=thinking_visible)
            if output is not None: return output
            continue # continue to the next iteration - left in case of changes later

    # Helper methods for generating payloads
    def gen_payload(self, history, tools, headers, max_tokens: int):
        if self.provider == "gemini":
            payload = self.gen_payload_gemini(history, tools, headers, max_tokens)
        
        elif self.provider == "anthropic":
            payload = self.gen_payload_anthropic(history, tools, headers, max_tokens)

        elif self.provider in ["openai", "custom"]:
            payload = self.gen_payload_openai_custom(history, tools, headers, max_tokens)

        elif self.provider == "ollama":
            payload = self.gen_payload_ollama(history, tools, headers, max_tokens)

        else:
            raise ValueError(f"Unsupported provider: {self.provider}. Please review how to configure custom providers.")
        return payload

    def gen_payload_gemini(self, history, tools, headers, max_tokens: int):
        if self.api_key: headers["x-goog-api-key"] = self.api_key
        payload = {"contents": history}
        if tools: payload["tools"] = tools
        if self.conversation.system_instruction:
            payload["systemInstruction"] = {"parts": [{"text": self.conversation.system_instruction}]}
        if max_tokens is not None:
            payload["generationConfig"] = {"maxOutputTokens": max_tokens}
        return payload
    
    def gen_payload_anthropic(self, history, tools, headers, max_tokens: int):
        if max_tokens is None: max_tokens = 4096
        headers.update({"x-api-key": self.api_key, "anthropic-version": "2023-06-01"})
        payload = {"model": self.model, "messages": history, "max_tokens": max_tokens}
        if tools: payload["tools"] = tools
        if self.conversation.system_instruction: payload["system"] = self.conversation.system_instruction
        return payload

    def gen_payload_openai_custom(self, history, tools, headers, max_tokens: int):
        if self.api_key: 
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {"model": self.model, "messages": history}
        if tools: 
            payload["tools"] = tools
        if max_tokens is not None:
            if self.model.startswith(("o1", "o3", "o4", "gpt-5")):
                payload["max_completion_tokens"] = max_tokens
            else:
                payload["max_tokens"] = max_tokens
        return payload
    
    def gen_payload_ollama(self, history, tools, headers, max_tokens: int):
        payload = {"model": self.model, "messages": history, "stream": False}
        if tools: payload["tools"] = tools
        if max_tokens is not None: payload["options"] = {"num_predict": max_tokens}
        return payload
    
    # Helper method for processing completed responses
    def process_response(self, res, thinking_visible=False):
        self._record_usage(res)  # every provider path funnels through here, sync or async

        if self.provider == "gemini":
            output = self.process_gemini_response(res, thinking_visible=thinking_visible)
            if output is not None: return output
            return None
        
        elif self.provider == "anthropic":
            output = self.process_anthropic_response(res, thinking_visible=thinking_visible)
            if output is not None: return output
            return None

        elif self.provider in ["openai", "custom"]:
            output = self.process_openai_custom_response(res, thinking_visible=thinking_visible)
            if output is not None: return output
            return None

        elif self.provider == "ollama":
            output = self.process_ollama_response(res, thinking_visible=thinking_visible)
            if output is not None: return output
            return None
        else:
            raise ValueError(f"Unsupported provider: {self.provider}. Please review how to configure custom providers.")

    def process_gemini_response(self, res, thinking_visible=False):
        if 'candidates' not in res or not res['candidates']:
            error_msg = res.get('error', {}).get('message', 'Unknown Gemini API Error')
            raise RuntimeError(f"Gemini API empty response or error: {error_msg}")

        parts = res['candidates'][0]['content']['parts']
        function_calls = [p['functionCall'] for p in parts if 'functionCall' in p]
        text_parts = [p['text'] for p in parts if 'text' in p]

        if function_calls:
            self._update_stage("Tool asked by AI: \n", str([f"   -{fc}" for fc in function_calls]), thinking_visible=thinking_visible)
            tool_calls = [{"id": function_call.get("id", ""),
                            "function": {"name": function_call['name'], "arguments": function_call.get('args', {})}}
                            for function_call in function_calls]
            self.conversation.add_model_msg(
                text="".join(text_parts) or None,
                tool_calls=tool_calls,
                native={"role": "model", "parts": parts},
                native_provider="gemini"
            )
            for function_call in function_calls:
                name, args = function_call['name'], function_call.get('args', {})
                call_id = function_call.get("id", "")
                
                self._update_stage("Running Tool Local Process", f"Invoking function '{name}'", thinking_visible=thinking_visible)
                out_str = self.tool_registry.execute(name, args)
                out_str = self.compression_policy.compress(out_str)

                function_response = {"name": name, "response": {"result": out_str}}
                if call_id: function_response["id"] = call_id
                self.conversation.history.append({
                    "role": "tool", "name": name, "content": out_str, "tool_call_id": call_id,
                    "_native": {"role": "user", "parts": [{"functionResponse": function_response}]},
                    "_native_provider": "gemini"
                })
            self._update_stage("Tool results sent to AI", thinking_visible=thinking_visible)
            return None
        
        text = "".join(text_parts)
        self.conversation.add_model_msg(text=text)
        self._update_stage("Success", thinking_visible=thinking_visible)
        return text

    def process_anthropic_response(self, res, thinking_visible=False):
        t_calls, final_text = [], ""
        for block in res.get("content", []):
            if block["type"] == "text": final_text += block["text"]
            elif block["type"] == "tool_use":
                t_calls.append({"id": block["id"], "function": {"name": block["name"], "arguments": block["input"]}})
        
        if t_calls:
            self._update_stage("Tool asked by AI: \n", str([f"   -{fc}" for fc in t_calls]), thinking_visible=thinking_visible)
            self.conversation.add_model_msg(
                text=final_text if final_text else None, tool_calls=t_calls,
                native={"role": "assistant", "content": res.get("content", [])}, native_provider="anthropic"
            )
            tool_result_blocks = []
            for tc in t_calls:
                self._update_stage("Running Tool Local Process", f"Invoking function '{tc['function']['name']}'", thinking_visible=thinking_visible)
                out_str = self.tool_registry.execute(tc["function"]["name"], tc["function"]["arguments"])
                out_str = self.compression_policy.compress(out_str)
                tool_result_blocks.append({"type": "tool_result", "tool_use_id": tc["id"], "content": out_str})
            
            native_user_msg = {"role": "user", "content": tool_result_blocks}
            for tc, block in zip(t_calls, tool_result_blocks):
                self.conversation.history.append({
                    "role": "tool", "name": tc["function"]["name"], "content": block["content"], "tool_call_id": tc["id"],
                    "_native": native_user_msg, "_native_provider": "anthropic"
                })
            self._update_stage("Tool results sent to AI", thinking_visible=thinking_visible)
            return None
        
        self.conversation.add_model_msg(text=final_text)
        self._update_stage("Success", thinking_visible=thinking_visible)
        return final_text

    def process_openai_custom_response(self, res, thinking_visible=False):
        msg = res['choices'][0]['message']
        if msg.get("tool_calls"):
            self._update_stage("Tool asked by AI: \n", str([f"   -{fc}" for fc in msg["tool_calls"]]), thinking_visible=thinking_visible)
            self.conversation.add_model_msg(text=msg.get("content"), tool_calls=msg["tool_calls"])
            for tool_call in msg["tool_calls"]:
                name = tool_call["function"]["name"]
                args = json.loads(tool_call["function"]["arguments"]) if isinstance(tool_call["function"]["arguments"], str) else tool_call["function"]["arguments"]
                
                self._update_stage("Running Tool Local Process", f"Invoking function '{name}'", thinking_visible=thinking_visible)
                out_str = self.tool_registry.execute(name, args)
                out_str = self.compression_policy.compress(out_str)
                self.conversation.add_tool_response(name, out_str, tool_call["id"])
            self._update_stage("Tool results sent to AI", thinking_visible=thinking_visible)
            return None
        
        text = msg.get("content", "")
        self.conversation.add_model_msg(text=text)
        self._update_stage("Success", thinking_visible=thinking_visible)
        return text    

    def process_ollama_response(self, res, thinking_visible=False):
        msg = res.get("message", {})
        if msg.get("tool_calls"):
            self._update_stage("Tool asked by AI: \n", str([f"   -{fc}" for fc in msg["tool_calls"]]), thinking_visible=thinking_visible)
            
            sanitized_calls = []
            for tc in msg["tool_calls"]:
                func = tc.get("function") or {}
                args = func.get("arguments") or {}
                if isinstance(args, str) and args.strip():
                    try: args = json.loads(args)
                    except json.JSONDecodeError: args = {}
                elif isinstance(args, str): args = {}
                sanitized_calls.append({
                    "id": tc.get("id"),
                    "function": {"name": func.get("name"), "arguments": args}
                })

            self.conversation.add_model_msg(text=msg.get("content"), tool_calls=sanitized_calls)
            for tc in sanitized_calls:
                name, args = tc["function"]["name"], tc["function"]["arguments"]
                self._update_stage("Running Tool Local Process", f"Invoking function '{name}'", thinking_visible=thinking_visible)
                out_str = self.tool_registry.execute(name, args)
                out_str = self.compression_policy.compress(out_str)
                self.conversation.add_tool_response(name, out_str, tc.get("id"))
            self._update_stage("Tool results sent to AI", thinking_visible=thinking_visible)
            return None
        
        text = msg.get("content", "")
        self.conversation.add_model_msg(text=text)
        self._update_stage("Success", thinking_visible=thinking_visible)
        return text

    # Other helper methods
    def change_default_max_tokens(self, max_tokens: int):
        self.max_tokens = max_tokens

    def set_system(self, instruction: str) -> "Agent":
        """Chainable method to update system instructions."""
        self.conversation.change_system_instruction(instruction)
        return self

    def set_max_tokens(self, tokens: int) -> "Agent":
        """Chainable method to update output token limits."""
        self.max_tokens = tokens
        return self

    def set_compression_policy(self, policy: CompressionPolicy) -> "Agent":
        """Chainable method to change how this agent compresses tool output before it hits
        history. Pass CompressionPolicy(enabled=False) to switch it back off."""
        self.compression_policy = policy
        return self

    def set_iterations(self, max_iterations: int) -> "Agent":
        """Chainable method to set tool execution limits."""
        self.set_max_tool_iterations(max_iterations)
        return self

    def with_tool(self, func: Callable, schema: dict = None) -> "Agent":
        """Chainable method to register a tool."""
        self.add_tool(func, schema)
        return self

    def spinner(self, spinner, thinking_visible, thread):
        # Keep the main thread alive to animate a spinner while waiting for network bytes
            spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
            spinner_idx = 0

            while thread.is_alive():
                if spinner:
                    print(f"\r[{self.provider.upper()}] ➔ Thinking {spinner_frames[spinner_idx]} ", end="", flush=True)
                    spinner_idx = (spinner_idx + 1) % len(spinner_frames)
                    time.sleep(0.08)
                else:
                    thread.join(timeout=0.1)  # Wait briefly to avoid busy-waiting
                    #self._update_stage(f"\r[{self.provider.upper()}] ➔ Thinking", thinking_visible=thinking_visible)
            
            # Clear the spinner trace line cleanly from the console terminal line
            print("\r" + " " * 50 + "\r", end="", flush=True)

    async def _animate_spinner(self, thinking_visible: bool):
        """Asynchronous terminal spinner running on the asyncio event loop."""
        spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        spinner_idx = 0
        try:
            while True:
                if thinking_visible:
                    print(f"\r[{self.provider.upper()}] ➔ Thinking {spinner_frames[spinner_idx]} ", end="", flush=True)
                    spinner_idx = (spinner_idx + 1) % len(spinner_frames)
                await asyncio.sleep(0.08)  # Relinquishes control cleanly to event loop
        except asyncio.CancelledError:
            # Clear spinner line cleanly when the network task completes
            print("\r" + " " * 50 + "\r", end="", flush=True)

    async def chat_async(self, user_input: str, custom_format_func: Callable[[list], list] = None, max_tokens: int = None,
                 thinking_visible: bool = True, spinner: bool = True) -> str:
            self.conversation.add_user_msg(user_input)
            
            return await self.generate_async(custom_format_func, max_tokens=max_tokens, thinking_visible=thinking_visible)
    
    async def generate_async(
        self, custom_format_func: Callable[[list], list] = None, 
        max_tokens: int = None, thinking_visible: bool = True, 
        spinner: bool = True) -> str:
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        self._update_stage("initial state", thinking_visible=thinking_visible)
        
        iterations = 0 
        while True:
            iterations += 1
            if iterations > self.max_tool_iterations:
                raise RuntimeError(f"Exceeded max_tool_iterations ({self.max_tool_iterations})")

            history = self.conversation.export_for(self.provider, special_format=custom_format_func)
            tools = self.tool_registry.export_for(self.provider, special_format=custom_format_func)
            headers = {"Content-Type": "application/json"}

            # --- 1. COMPILE THE TARGET PAYLOAD ---
            payload = self.gen_payload(history, tools, headers, max_tokens)

            # --- 2. SEND THE REQUEST ---
            self._update_stage("Request sent to AI", f"Payload Size: {len(history)} turns", thinking_visible=thinking_visible)

            spinner_async_task = None
            if spinner:
                spinner_async_task = asyncio.create_task(self._animate_spinner(thinking_visible))

            response = None
            try:
                # Offloads urllib request to thread pool; returns result directly or raises exception
                response = await asyncio.to_thread(self._send_request, payload, headers)
            except Exception as err:
                # Preserves explicit error logging before bubbling up
                self._update_stage("Network/API Exception", str(err), thinking_visible=thinking_visible)
                raise err
            finally:
                if spinner_async_task:
                    spinner_async_task.cancel()
                    try:
                        await spinner_async_task
                    except asyncio.CancelledError:
                        pass

            # --- 3. PROCESS COMPLETED RESPONSE OBJECTS ---
            output = self.process_response(response, thinking_visible=thinking_visible)
            if output is not None: 
                return output
            continue # continue to the next iteration - left in case of changes later


def test():
    import dotenv
    dotenv.load_dotenv()
    agent = Agent('gemini', 'gemini-3.5-flash')
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break
        response = agent.chat(user_input)
        print(f"Agent: {response}")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break
        response = agent.chat(user_input, thinking_visible=False, spinner=False)
        print(f"Agent: {response}")
    
    agent = Agent('ollama', 'llama3')
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break
        response = agent.chat(user_input)
        print(f"Agent: {response}")

if __name__ == "__main__":
    test()
