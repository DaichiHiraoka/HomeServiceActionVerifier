from __future__ import annotations

from pathlib import Path


def test_new_package_import_and_cli_parser() -> None:
    import home_service_action_verifier
    from home_service_action_verifier.cli import build_parser

    parser = build_parser()

    assert home_service_action_verifier.__version__ == "0.1.0"
    assert parser.prog == "home-service-verifier"


def test_old_source_package_directory_removed() -> None:
    assert not Path("src/privacy_vlm_poc").exists()
