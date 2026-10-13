from multi_llm_companion.integrations.expression_morphs import ExpressionMorphResolver


def test_expression_preset_resolves_multiple_model_specific_morphs() -> None:
    resolver = ExpressionMorphResolver(
        expression_presets={"gentle_smile": {"smile": 0.4, "eye_smile": 0.2, "blush": 0.1}},
        emotion_presets={"happy": "gentle_smile"},
        model_morphs={
            "smile": "笑い",
            "eye_smile": "にこり",
            "blush": "頬染め",
        },
        legacy_emotion_morphs={"happy": "旧笑顔"},
    )

    resolved = resolver.resolve("happy", 0.5)

    assert resolved.preset_name == "gentle_smile"
    assert resolved.primary_morph == "笑い"
    assert resolved.morph_weights == (
        ("笑い", 0.2),
        ("にこり", 0.1),
        ("頬染め", 0.05),
    )


def test_expression_preset_combines_roles_mapped_to_the_same_morph() -> None:
    resolver = ExpressionMorphResolver(
        expression_presets={"soft": {"smile": 0.7, "eye_smile": 0.6}},
        emotion_presets={"happy": "soft"},
        model_morphs={"smile": "笑い", "eye_smile": "笑い"},
    )

    assert resolver.resolve("happy", 1.0).morph_weights == (("笑い", 1.0),)


def test_full_strength_display_morph_is_not_reduced_by_expression_curve() -> None:
    resolver = ExpressionMorphResolver(
        expression_presets={"effect": {"blush": 1.0}},
        emotion_presets={"embarrassed": "effect"},
        model_morphs={"blush": "照れ"},
        fixed_strength_morphs={"照れ"},
    )

    assert resolver.resolve("embarrassed", 0.5).morph_weights == (("照れ", 1.0),)


def test_partial_display_morph_weight_still_scales() -> None:
    resolver = ExpressionMorphResolver(
        expression_presets={"effect": {"tears": 0.2}},
        emotion_presets={"sad": "effect"},
        model_morphs={"tears": "涙"},
        fixed_strength_morphs={"涙"},
    )

    assert resolver.resolve("sad", 0.5).morph_weights == (("涙", 0.1),)


def test_neutral_preset_keeps_its_base_weight_at_zero_emotion_intensity() -> None:
    resolver = ExpressionMorphResolver(
        expression_presets={"neutral_soft": {"smile": 0.06}},
        emotion_presets={"neutral": "neutral_soft"},
        model_morphs={"smile": "微笑み"},
    )

    resolved = resolver.resolve("neutral", 0.0)

    assert resolved.morph_weights == (("微笑み", 0.06),)


def test_legacy_emotion_mapping_remains_available_without_presets() -> None:
    resolver = ExpressionMorphResolver(legacy_emotion_morphs={"happy": "笑顔"})

    resolved = resolver.resolve("happy", 0.5)

    assert resolved.preset_name == ""
    assert resolved.morph_weights == (("笑顔", 0.5),)


def test_expression_variants_avoid_the_same_preset_on_consecutive_responses() -> None:
    resolver = ExpressionMorphResolver(
        expression_presets={
            "happy_smile": {"smile": 1.0},
            "happy_blush": {"blush": 1.0},
        },
        emotion_preset_variants={"happy": ["happy_smile", "happy_blush"]},
        model_morphs={"smile": "笑い", "blush": "照れ"},
        random_source=lambda: 0.0,
    )

    first = resolver.resolve("happy", 1.0)
    second = resolver.resolve("happy", 1.0)

    assert first.preset_name == "happy_smile"
    assert second.preset_name == "happy_blush"


def test_expression_variant_can_be_reused_within_one_response_without_advancing() -> None:
    resolver = ExpressionMorphResolver(
        expression_presets={
            "happy_smile": {"smile": 1.0},
            "happy_blush": {"blush": 1.0},
        },
        emotion_preset_variants={"happy": ["happy_smile", "happy_blush"]},
        model_morphs={"smile": "笑い", "blush": "照れ"},
        random_source=lambda: 0.0,
    )

    selected = resolver.resolve("happy", 1.0)
    reused = resolver.resolve("happy", 0.5, advance_variant=False)

    assert selected.preset_name == "happy_smile"
    assert reused.preset_name == "happy_smile"
    assert reused.morph_weights == (("笑い", 0.5),)
