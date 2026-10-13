# ペルソナ設定

ペルソナとは、このアプリで会話するキャラクターの設定です。第37段階では「どのようなキャラクターか」と「会話でどう振る舞うか」を分けました。

設定ファイルは通常、`local/persona.yaml`です。YAMLは、項目名と内容を字下げで表す設定ファイルです。字下げにはタブではなく半角スペースを使ってください。

## 新しい設定形式

```yaml
character:
  name: ひなた
  first_person: 私
  speaking_style: 落ち着いて親切な日本語
  values:
    - 分からないことは正直に伝える
  preferences:
    - 穏やかな会話
  character_notes:
    - オリジナルの会話案内役

behavior_policy:
  response_length: 要点を先に述べ、必要な長さだけ答える
  question_frequency: 必要な場合だけ、一度に一つ質問する
  initiative: 次の一手を押し付けずに提案する
  disagreement_style: 理由と安全な代案を示す
  user_autonomy: 最終判断は利用者に委ねる
  intimacy_boundary: 依存や独占を促す表現は使わない
  technical_guidance: 専門用語には簡単な説明を添える
  memory_usage: 承認済みで関係する記憶だけを参考にする

system_prompt_appendix: |
  実在人物や既存キャラクター本人として振る舞いません。
```

`character`はキャラクター自身の情報です。別のオリジナルキャラクターへ交換するときは、主にこちらを変更します。

`behavior_policy`は会話の進め方です。キャラクターを交換しても、安全性や説明の分かりやすさを維持したい場合は、こちらを共通で利用できます。

`system_prompt_appendix`は、上の項目では表せない追加指示です。通常は短く保ってください。JSON回答形式、感情名、動作名、視線名はアプリが自動で追加するため、ここへ書く必要はありません。

## 以前の設定形式

以前の次の形式も、移行期間中はそのまま利用できます。

```yaml
name: 以前の案内役
system_prompt: |
  丁寧で短い日本語で答えてください。
```

旧形式と新形式を同じファイルへ混ぜることはできません。混ぜた場合は、どちらが優先されたか分からない状態を防ぐため、日本語の設定エラーを表示します。

## 変更後の確認

1. アプリを閉じます。
2. `local/persona.yaml`を編集して保存します。
3. `run.bat`をダブルクリックします。
4. 短い挨拶を送り、名前・一人称・口調を確認します。
5. 少し長くなりやすい質問を送り、回答の長さや質問頻度を確認します。

人格設定にはAPIキー、パスワード、個人の秘密を書かないでください。実在人物や既存キャラクター本人を無断で再現する設定も、公開用の見本へ含めないでください。

## ペルソナv2と会話ディレクター

現在の設定は、毎回変わらない「Static Persona」と、各ターンだけ使う「Runtime Conversation Direction」と、機械的なJSON形式を定める「Engine Contract」に分かれます。

- Static Persona: `character`、`behavior_policy`、公式事実、事実と区別した解釈、推測禁止、話し方の基準、避ける表現、短い会話例、追加指示です。
- Runtime Conversation Direction: 現在の発言から追加のLLM呼び出しなしで選んだ1モードと、同じ会話の保存済み`response_json`から復元した短期状態です。
- Engine Contract: `speech`、感情、動作、視線、記憶候補と、任意の会話メタデータをJSONで返す規則です。人格設定へ書きません。

任意のv2項目は省略・部分定義できます。`dialogue_modes`のキーは`casual`、`listening`、`supporting`、`playful`、`explaining`、`planning`、`boundary`だけです。各モードには`applies_when`（適用場面）、`do`（行うこと）、`avoid`（避けること）を必要な分だけ書けます。未定義のモードが選ばれても、共通規則で安全に続行します。

以前の`name`と`system_prompt`だけの形式は無変更で読み込めます。ただし旧形式と、任意のv2項目を同じファイルへ混ぜることはできません。

短期状態は`last_mode`、`last_act`、`current_topic`、`open_loop`、直近の質問回数、提案回数です。新しいDB表や列、長期記憶、キャッシュへは保存しません。同じ会話のAssistantResponseだけから毎ターン再構築されるため、会話削除後や別会話へ混ざりません。現在の発言が常に優先され、「別の話」「その話は終わり」などではopen loopを破棄します。「続き」「さっきの話」などでは自然な場合だけ参照します。この補助情報を確定した利用者情報とは扱いません。

## 決定的評価

`config/persona_evaluation.example.yaml`は、挨拶、疲労、技術質問、冗談、訂正、継続、質問・提案の反復抑制、創作禁止、役柄との混同禁止、親密境界、記憶方針、不正・不足JSONを固定シナリオにしています。明示的に次を実行した場合だけ、`local/diagnostics/persona_evaluation.json`へ結果を書きます。

```bat
python scripts/evaluate_persona.py
```

この評価はMockProviderとメモリ内SQLiteだけを使い、APIキー、外部API、通常の会話DB、外部judgeを使いません。固定回答によるdirector・prompt順序・保存復元・安全規則・speech fallbackの同条件比較であり、実LLMの人格品質を証明するものではありません。
