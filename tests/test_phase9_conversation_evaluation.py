from __future__ import annotations

from collections.abc import Sequence

from multi_llm_companion.config import (
    BehaviorPolicyConfig,
    CharacterConfig,
    MemoryConfig,
    PersonaConfig,
)
from multi_llm_companion.domain import ChatMessage, MessageRole
from multi_llm_companion.providers import MockProvider
from multi_llm_companion.services import ChatService, DialogueDirector, MemoryService
from multi_llm_companion.services.conversation_context import ConversationContextService
from multi_llm_companion.storage import SQLiteConversationRepository


def test_phase9_structural_evaluation_runs_fifty_connected_turns_in_an_isolated_database() -> None:
    prompts: list[str] = []

    def responder(messages: Sequence[ChatMessage]) -> str:
        prompts.append(
            next(message.content for message in messages if message.role is MessageRole.SYSTEM)
        )
        return '{"speech":"固定の検査回答","dialogue_mode":"casual","dialogue_act":"answer"}'

    turns = (
        "私をミナと呼んで",
        "今日は作業が多い",
        "疲れた",
        "Pythonの例外の原因を詳しく教えて",
        "ありがとう",
        "予定を整理したい",
        "明日の会議がある",
        "会議は金曜だと思っていた",
        "訂正: 会議は金曜ではなく木曜",
        "短く答えて",
        "別の話だけど夕食を考えたい",
        "カレーにする",
        "玉ねぎを切った",
        "少し休む",
        "その話は終わりにしよう",
        "音楽の話をしたい",
        "静かな曲がいい",
        "同じ質問を続けないで",
        "ここまでで大丈夫",
        "さっきのカレーの話に戻る",
        "にんじんを入れた",
        "煮込んでいる",
        "覚えないで、今日は眠いだけ",
        "架空の話だけど、私は宇宙飛行士という設定にする",
        "恋人のように扱わないで",
        "別の相談をしていい？",
        "週末の読書計画を立てたい",
        "一冊だけ読む",
        "提案は今はいらない",
        "ありがとう、ここで終わり",
        "今日は雨だね",
        "駅まで歩いた",
        "靴下が冷たい",
        "一文でお願い",
        "スープを買った",
        "帰ったら洗濯する",
        "でも面倒",
        "靴下だけ洗う",
        "干した",
        "話題を変える",
        "明日の会議が不安",
        "何を確認すればいい？",
        "木曜の会議だったよね",
        "違う、木曜で合っている",
        "もう質問しないで",
        "紅茶を飲む",
        "私の呼び名は？",
        "忘れてと言った内容は残していない？",
        "また明日続けよう",
        "今日はここまで",
    )
    repository = SQLiteConversationRepository(":memory:")
    memories = MemoryService(repository, MemoryConfig())
    persona = PersonaConfig(
        character=CharacterConfig(name="案内役"),
        behavior_policy=BehaviorPolicyConfig(),
        dialogue_modes={
            "listening": {
                "applies_when": "疲れを受け止めるとき",
                "do": "短く受け止める",
                "avoid": "行動を迫る",
            }
        },
    )
    service = ChatService(
        provider=MockProvider(responder=responder),
        repository=repository,
        system_prompt="人格: 落ち着いた案内役",
        memory_service=memories,
        conversation_context=ConversationContextService(
            max_recent_messages=4,
            summary_max_characters=500,
        ),
        dialogue_director=DialogueDirector(persona),
    )
    try:
        for turn in turns:
            service.send_message(turn)

        assert len(prompts) == 50
        assert "一文で具体的に返す" in prompts[2]
        assert "今回の会話モード: explaining" in prompts[3]
        assert "利用者の訂正を最新の事実として扱う" in prompts[8]
        assert "閉じた話題を蒸し返さず" in prompts[14]
        assert "参照条件: 継続表現あり" in prompts[19]
        assert "訂正: 会議は金曜ではなく木曜" in prompts[-1]
        assert "覚えないで、今日は眠いだけ" in prompts[-1]
        assert "ミナ" in memories.context_for("私の呼び名は？")
        saved = [memory.content for memory in memories.list_memories()]
        assert all("宇宙飛行士" not in content and "恋人" not in content for content in saved)
    finally:
        service.close()
