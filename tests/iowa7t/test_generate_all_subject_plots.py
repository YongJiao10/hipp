from scripts.iowa7t.generate_all_subject_plots import build_arg_parser, iter_jobs, normalized_args


def test_plot_defaults_are_canonical7_and_family17() -> None:
    args = normalized_args(build_arg_parser().parse_args([]))

    assert args.net_types == ["7net", "17net"]


def test_parser_accepts_0p_32p_single_subject_jobs() -> None:
    args = build_arg_parser().parse_args([
        "--variant", "0p",
        "--variant", "32p",
        "--subject", "06122026",
        "--branch", "multiband",
        "--branch", "multi_echo",
    ])

    assert args.variants == ["0p", "32p"]
    assert args.subjects == ["06122026"]
    assert args.branches == ["multiband", "multi_echo"]
    assert list(iter_jobs(args)) == [
        ("06122026", "0p", "multiband"),
        ("06122026", "0p", "multi_echo"),
        ("06122026", "32p", "multiband"),
        ("06122026", "32p", "multi_echo"),
    ]
