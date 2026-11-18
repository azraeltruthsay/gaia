"""42qa: routes on a prefixed router (APIRouter(prefix=...), or mounted with
include_router(..., prefix=...)) read as missing; 96 of 119 'mismatches'."""
from gaia_common.utils.blueprint_precheck import _find_http_endpoint, _find_ws_endpoint

ROUTER = 'from fastapi import APIRouter\nrouter = APIRouter(prefix="/sleep", tags=["sleep"])\n\n@router.post("/wake")\ndef wake(): ...\n'
SUB = 'router = APIRouter()\n\n@router.get("")\ndef log(): ...\n\n@router.api_route("/wiki/{path:path}", methods=["GET", "HEAD"])\ndef w(): ...\n'
MAIN = ('from svc.routes.changelog import router as changelog_router\n'
        'app.include_router(changelog_router, prefix="/api/changelog", tags=["c"])\n')


def test_router_prefix_applies():
    assert _find_http_endpoint("POST", "/sleep/wake", {"api/sleep.py": ROUTER})[0] == "api/sleep.py"
    assert _find_http_endpoint("POST", "/wake", {"api/sleep.py": ROUTER})[0] is None


def test_mount_prefix_and_empty_path():
    files = {"main.py": MAIN, "routes/changelog.py": SUB}
    assert _find_http_endpoint("GET", "/api/changelog", files)[0] == "routes/changelog.py"


def test_api_route_methods():
    files = {"routes/wiki.py": SUB}
    assert _find_http_endpoint("HEAD", "/wiki/{path}", files)[0] == "routes/wiki.py"
    assert _find_http_endpoint("POST", "/wiki/{path}", files)[0] is None


def test_websocket_on_app():
    assert _find_ws_endpoint("/ws", {"m.py": '@app.websocket("/ws")\nasync def ws(): ...\n'})[0] == "m.py"
