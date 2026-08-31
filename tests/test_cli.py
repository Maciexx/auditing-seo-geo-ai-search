from ai_search_audit.cli import build_parser


def test_cli_supports_audit_command() -> None:
    args = build_parser().parse_args(["audit", "example.com", "--output-dir", "out"])
    assert args.command == "audit"
    assert args.domain == "example.com"
    assert str(args.output_dir) == "out"


def test_cli_supports_documented_output_alias() -> None:
    parser = build_parser()
    audit_parser = parser._subparsers._group_actions[0].choices["audit"]
    output_action = next(action for action in audit_parser._actions if action.dest == "output_dir")
    assert {"--output", "--output-dir"} <= set(output_action.option_strings)
    args = parser.parse_args(
        ["audit", "https://lakeside-hotel.example", "--output", "audit-output/sample"]
    )
    assert str(args.output_dir) == "audit-output/sample"


def test_cli_supports_polish_report_locale() -> None:
    args = build_parser().parse_args(["audit", "example.com", "--report-locale", "pl"])

    assert args.report_locale == "pl"
