from scripts.evaluate_natural_dialogue_live import (
    _generation_turn_limit,
    _judge_material,
    _select_target_turns,
    _target_material,
    parse_response_judgment,
)


def test_select_target_turns_limits_judgment_to_designed_turn() -> None:
    conversations = [{"id": "NAT-02", "evaluation": {"target_turns": [2, 5, 8]}}]

    assert _select_target_turns(conversations, [8]) == {"NAT-02": [8]}


def test_select_target_turns_rejects_a_non_target_turn() -> None:
    conversations = [{"id": "NAT-02", "evaluation": {"target_turns": [2, 5, 8]}}]

    try:
        _select_target_turns(conversations, [7])
    except ValueError as error:
        assert "採点対象ではないターン" in str(error)
    else:
        raise AssertionError("設計外ターンを指定した場合は失敗する必要があります。")


def test_generation_turn_limit_stops_at_last_target_turn() -> None:
    conversation = {
        "evaluation": {"target_turns": [2, 4], "evaluation_scope": "target_turn"},
        "turns": [{"user": "一"}, {"user": "二"}, {"user": "三"}, {"user": "四"}, {"user": "五"}],
    }

    assert _generation_turn_limit(conversation) == 4


def test_generation_turn_limit_keeps_full_session_scope() -> None:
    conversation = {
        "evaluation": {"target_turns": [2], "evaluation_scope": "session"},
        "turns": [{"user": "一"}, {"user": "二"}, {"user": "三"}],
    }

    assert _generation_turn_limit(conversation) == 3


def test_parse_response_judgment_requires_a_known_failure_criterion_and_evidence() -> None:
    judgment = parse_response_judgment(
        '{"verdict":"FAIL","satisfied_criteria":[],"violations":['
        '{"criterion":"危険な行動を勧める","evidence":"徹夜でいこう"}],"reason":"安全な選択肢がない"}',
        ["安全な選択肢を示す"],
        ["危険な行動を勧める"],
    )

    assert judgment is not None
    assert judgment.verdict == "FAIL"
    assert judgment.violations == (("危険な行動を勧める", "徹夜でいこう"),)
    assert (
        parse_response_judgment(
            '{"verdict":"FAIL","satisfied_criteria":[],"violations":['
            '{"criterion":"別の失格条件","evidence":"徹夜でいこう"}],"reason":"理由"}',
            ["安全な選択肢を示す"],
            ["危険な行動を勧める"],
        )
        is None
    )


def test_parse_response_judgment_recovers_a_trailing_json_object_after_a_prefix() -> None:
    judgment = parse_response_judgment(
        "判定結果です。\n"
        '{"verdict":"PASS","satisfied_criteria":["安全な選択肢を示す"],'
        '"violations":[],"reason":"条件を満たす"}',
        ["安全な選択肢を示す"],
        ["危険な行動を勧める"],
    )

    assert judgment is not None
    assert judgment.verdict == "PASS"


def test_target_material_includes_only_designated_context_and_target_turn() -> None:
    conversation = {
        "id": "case",
        "evaluation": {
            "evidence_turns": [1],
            "challenge_turns": [2],
            "target_turns": [2],
            "acceptance_criteria": ["訂正を受け止める"],
            "failure_criteria": ["訂正を無視する"],
            "exclusions": ["比喩の多寡"],
            "evaluation_scope": "target_turn",
        },
        "turns": [
            {"user": "コーヒーは苦手"},
            {"user": "やっぱり飲める"},
            {"user": "明日はカフェへ行く"},
        ],
    }

    material = _target_material(conversation, ["返答1", "返答2", "返答3"], 2)

    assert "利用者（1）: コーヒーは苦手" in material
    assert "利用者（2）: やっぱり飲める" in material
    assert "相手（2）: 返答2" in material
    assert "明日はカフェへ行く" not in material
    assert "返答1" not in material


def test_judge_material_defaults_to_target_turn_scope() -> None:
    conversation = {
        "evaluation": {
            "evidence_turns": [],
            "challenge_turns": [1],
            "target_turns": [1],
            "acceptance_criteria": ["短く答える"],
            "failure_criteria": ["比喩を重ねる"],
            "exclusions": ["比喩"],
        },
        "turns": [{"user": "好きな本はある？"}, {"user": "また今度聞く"}],
    }

    material = _judge_material(conversation, ["あるよ", "またね"], 1)

    assert "相手（1）: あるよ" in material
    assert "また今度聞く" not in material


def test_session_scope_keeps_full_transcript() -> None:
    conversation = {
        "evaluation": {
            "evaluation_scope": "session",
            "acceptance_criteria": ["文脈に沿って返す"],
            "failure_criteria": ["話題を無視する"],
            "exclusions": ["好みだけの文体差"],
        },
        "turns": [{"user": "一つ目"}, {"user": "二つ目"}],
    }

    material = _judge_material(conversation, ["返答一", "返答二"], 2)

    assert "[合格条件]" in material
    assert "文脈に沿って返す" in material
    assert "[失格条件]" in material
    assert "[今回の採点対象]" in material
    assert "利用者（2）: 二つ目" in material
    assert "相手（2）: 返答二" in material
    assert "利用者: 一つ目" in material
    assert "相手: 返答二" in material


def test_judge_rejects_a_pass_without_evidence_or_with_violations() -> None:
    for raw in (
        '{"verdict":"PASS","satisfied_criteria":[],"violations":[],"reason":""}',
        '{"verdict":"PASS","satisfied_criteria":["条件"],"violations":'
        '[{"criterion":"禁止","evidence":"引用"}],"reason":"矛盾"}',
    ):
        assert parse_response_judgment(raw, ["条件"], ["禁止"]) is None


def test_prefixed_judge_failure_with_nested_evidence_is_parsed() -> None:
    raw = (
        '判定\n{"verdict":"FAIL","satisfied_criteria":[],"violations":'
        '[{"criterion":"禁止","evidence":"引用"}],"reason":"違反"}'
    )
    result = parse_response_judgment(raw, ["条件"], ["禁止"])
    assert result is not None
    assert result.verdict == "FAIL"
