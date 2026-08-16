"""Optional real-browser smoke test for the Phase 8 GUI.

NOT part of the suite and NOT run by tests/run_all.py (its filename
deliberately doesn't match the test_* pattern), because it needs playwright -
a third-party dependency llmadapt does not have and should not acquire.
tests/test_gui.py covers the server, the API and the page's structure without
a browser; this file covers the part that one honestly cannot: the DOM
behaviour.

Run it when changing gui_assets.py:

    pip install playwright && playwright install chromium
    PYTHONPATH=src python3 tests/browser_smoke.py

It drives a headless Chromium through the interactions that have no other
coverage - loading a template, adding and renaming an employee, dragging a
node, connect mode, the right-click bulk-action menu, the reporting-cycle
guard - and fails on any console error.
"""

import sys

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    print("SKIP: playwright is not installed - see this file's docstring.")
    sys.exit(0)

from llmadapt.gui import launch_gui


def main():
    server = launch_gui(open_browser=False, block=False)
    errors = []

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1400, "height": 900})
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(server.url)
        page.wait_for_function("document.querySelectorAll('#palette option').length > 0", timeout=10000)

        assert page.eval_on_selector_all("#palette option", "e => e.length") >= 4
        assert page.eval_on_selector_all("#template option", "e => e.length") >= 5
        print("PASS: the page populates its controls from /api/options")

        page.select_option("#template", "small-coding-team")
        page.select_option("#size", "medium")
        page.click("#applyTemplate")
        page.wait_for_function("document.querySelectorAll('g.node').length >= 5", timeout=10000)
        assert page.eval_on_selector_all("path.edge", "e => e.length") >= 4
        print("PASS: loading a template draws its nodes and reporting edges")

        page.click("#addBtn")
        page.wait_for_timeout(200)
        name_input = page.query_selector("#editor input")
        name_input.fill("Newbie")
        name_input.dispatch_event("change")
        page.wait_for_timeout(200)
        labels = page.eval_on_selector_all("g.node text.nm", "els => els.map(e => e.textContent)")
        assert "Newbie" in labels, labels
        print("PASS: an employee can be added and renamed from the side panel")

        before = page.evaluate("state.spec.layout['Newbie'].slice()")
        box = page.query_selector("g.node[data-name='Newbie'] rect").bounding_box()
        page.mouse.move(box["x"] + 40, box["y"] + 20)
        page.mouse.down()
        page.mouse.move(box["x"] + 240, box["y"] + 180, steps=8)
        page.mouse.up()
        page.wait_for_timeout(200)
        after = page.evaluate("state.spec.layout['Newbie'].slice()")
        assert abs(after[0] - before[0]) > 100, (before, after)
        print("PASS: nodes drag, and the layout follows them")

        page.click("#connectBtn")
        page.click("g.node[data-name='Newbie'] rect")
        page.click("g.node[data-name='Manager'] rect")
        page.wait_for_timeout(200)
        assert page.evaluate(
            "state.spec.employees.find(e => e.name==='Newbie').reports_to") == "Manager"
        print("PASS: connect mode wires a reporting line by clicking two nodes")

        page.click("#canvas", button="right", position={"x": 900, "y": 700})
        page.wait_for_timeout(150)
        items = page.eval_on_selector_all("#menu button", "els => els.map(e => e.textContent)")
        assert any("Connect all" in i for i in items), items
        page.click("#menu button:has-text('Select all')")
        page.wait_for_timeout(150)
        assert page.evaluate("state.selected.size") >= 6
        print("PASS: the right-click menu offers the bulk actions and 'Select all' works")

        page.evaluate("connect('Chief','Developer 1')")
        page.wait_for_timeout(100)
        page.evaluate("connect('Developer 1','Chief')")
        assert page.evaluate("state.spec.employees.find(e=>e.name==='Chief').reports_to") is None
        print("PASS: the reporting-cycle guard refuses to close a loop")

        page.click("#check")
        page.wait_for_timeout(400)
        assert "valid" in page.text_content("#status").lower()
        page.click("#build")
        page.wait_for_function(
            "document.body.innerHTML.includes('handed back to Python')", timeout=10000)
        print("PASS: check reports a valid design and build hands it back")
        browser.close()

    assert server.done.wait(5), "build should release the waiting caller"
    assert server.result.ok
    assert errors == [], errors
    print(f"\nBrowser smoke test passed - built {len(server.result.company.employees)} employees, "
          f"no console errors.")


if __name__ == "__main__":
    main()
