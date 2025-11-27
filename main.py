import os
import json
import streamlit as st
import openai

# API配置

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_API_BASE", "https://api.cursorai.art/v1")
MODEL_NAME = "gpt-5.1-thinking-all"
if not openai.api_key:
    raise RuntimeError("未找到环境变量 CURSOR_API_KEY，请先配置你的 API 密钥。")

def call_chat_llm(system_prompt: str, user_content: str, temperature: float = 0.2) -> str:
    """
    使用 openai==0.28 的 ChatCompletion 接口，走你配置的 api_base 代理。
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_content},
    ]

    try:
        resp = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=temperature,
        )
        return resp.choices[0].message["content"]
    except Exception as e:

        return f"[调用模型出错: {e}]"





def safe_parse_json(text: str, fallback: dict):
    """尽量从 LLM 输出中提取 JSON，对 ```json 包裹等做清理。"""
    try:
        text = text.strip()
        if text.startswith("```"):
            first_brace = text.find("{")
            last_brace = text.rfind("}")
            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                text = text[first_brace:last_brace + 1]

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start:end + 1]

        return json.loads(text)
    except Exception:
        return fallback


# ========= 配置：6 类话题 =========
TOPIC_TYPES = [
    {"id": 1, "label": "童年与成长"},
    {"id": 2, "label": "青年 / 成家立业"},
    {"id": 3, "label": "工作与成就"},
    {"id": 4, "label": "旅行与地点记忆"},
    {"id": 5, "label": "兴趣 / 技能 / 日常生活"},
    {"id": 6, "label": "轻松 / 中性话题"},
]


# ========= 1. 话题分类 =========
TOPIC_CLASSIFIER_SYSTEM = """
你是一个中文话题分类器。

现在用户会说一段话，你需要判断这段话主要属于哪一类话题。
系统共有 6 类话题：

1. 童年与成长：童年回忆、成长经历、校园生活、小学/中学时代的故事等；
2. 青年 / 成家立业：大学、初入社会、恋爱与结婚、成家、生育、人生重大选择等；
3. 工作与成就：工作压力、职业发展、职场人际、绩效、科研/学术成就、个人目标与理想等；
4. 旅行与地点记忆：旅游经历、某个城市/地点相关的记忆、出差见闻、在某地生活的体验等；
5. 兴趣 / 技能 / 日常生活：兴趣爱好、学习技能、日常习惯、生活琐事、娱乐活动、宠物等；
6. 轻松 / 中性话题：不带明显强烈情绪的聊天内容、随意闲聊、吐槽但情绪不强烈等。

你会收到一个 JSON 对象，包含：
- user_text: 用户当前这次输入的文本（可能包含前后文补充）
- rejected_topic_ids: 一个整数列表，表示哪些话题类型编号已经被用户明确否定过
- rejected_topic_labels: 这些编号对应的话题名称

你的任务：
1. 根据 user_text 在 1~6 中选出最贴近的一类；
2. 如果 rejected_topic_ids 不为空，说明用户已经明确表示“这些类型不对”，
   正常情况下不要再输出这些类型；
3. 只有在极端情况下（例如 rejected_topic_ids 已经包含 1~6 的全部），
   你才可以在这些被拒绝的类型中选一个“最不差的”类型作为输出；
4. 无论如何，必须输出一个合法的 topic_id（1~6）。

只输出一个 JSON，不要多余解释，格式如下：

{
  "topic_id": 2,
  "topic_label": "青年 / 成家立业",
  "reason": "..."
}
"""

def classify_topic(user_text: str, rejected_topic_ids=None) -> dict:
    if rejected_topic_ids is None:
        rejected_topic_ids = []

    # 根据 id 找到对应的标签
    rejected_labels = [
        t["label"] for t in TOPIC_TYPES
        if t["id"] in rejected_topic_ids
    ]

    payload = {
        "user_text": user_text,
        "rejected_topic_ids": rejected_topic_ids,
        "rejected_topic_labels": rejected_labels,
    }

    fallback = {
        "topic_id": 6,
        "topic_label": "轻松 / 中性话题",
        "reason": "无法确定，更像是比较轻松或中性的聊天内容"
    }

    raw = call_chat_llm(
        TOPIC_CLASSIFIER_SYSTEM,
        json.dumps(payload, ensure_ascii=False)
    )
    data = safe_parse_json(raw, fallback)
    if "topic_id" not in data or "topic_label" not in data:
        return fallback
    return data



# ========= 2. 槽位抽取：who / what / where / when / why / how =========
SLOT_EXTRACTOR_SYSTEM = """
你是一个信息抽取器，负责从对话中抽取六个要素：

- who: 主要涉及到的关键人物（用户自己 + 其他重要人物）；
- what: 发生的事情 / 核心事件；
- where: 发生地点（如果没有提到，就留空 null）；
- when: 时间点或时间段（例如“去年暑假”“上个月”“这两年”“2023 年国庆期间”等）；
- why: 事件发生的原因、动机或背景；
- how: 用户或他人采取了哪些行动 / 现在打算怎么做，包括大致过程、准备、决策方式等。

你会收到：
1）当前完整对话（多轮）；
2）当前已记录的 slots（可能有部分内容）。

【非常重要的约束】：
1. 只能根据用户**明确说出口的内容**抽取信息，不能凭常识或猜测补全。
   - 不允许输出“可能是……”“大概是为了……”这类推断性话语。
   - 如果用户没有明确说明 why / how，就把它们保持为 null，而不是自己猜一个合理的理由。
2. when 表示事件发生的时间点或时间段。
   - 像“花了两个半月的时间”“大概玩了三周”这类只是**持续时长**，不要填入 when。
   - 如果只提到持续时长而没有时间点，则 when 设为 null，可以把“持续了多久”放在 how 里。
3. what / why / how 如果内容非常笼统（例如 “去国外玩了一圈”“就是想出去走走”），
   也可以先记录下来，但后续一旦用户补充了更具体的事件或原因，要用更详细的描述覆盖之前的值。
4. 如果确实没有足够信息，请毫不犹豫地用 null，而不是勉强编一句话填充。

请你只输出一个 JSON，不要多余解释，例如：

{
  "who": "我和室友小张",
  "what": "昨天晚上因为打游戏的声音太大吵了一架",
  "where": "宿舍里",
  "when": "昨天晚上十点左右",
  "why": "他觉得我影响他休息，而我觉得他太敏感",
  "how": "我当时直接怼回去，后来就不说话了，现在想看看怎么缓和关系"
}
"""

def extract_slots(history, current_slots: dict) -> dict:
    convo_text = ""
    for msg in history:
        role = "用户" if msg["role"] == "user" else "助手"
        convo_text += f"{role}: {msg['content']}\n"

    user_content = json.dumps({
        "conversation": convo_text,
        "current_slots": current_slots,
    }, ensure_ascii=False)

    fallback = current_slots.copy()
    raw = call_chat_llm(SLOT_EXTRACTOR_SYSTEM, user_content)
    data = safe_parse_json(raw, fallback)

    for k in ["who", "what", "where", "when", "why", "how"]:
        if k not in data:
            data[k] = current_slots.get(k)
    return data


def is_filled_val(v, key=None):
    """根据不同要素使用不同的“够详细”的阈值。"""
    if v is None or not isinstance(v, str):
        return False
    text = v.strip()
    if not text:
        return False

    # 默认门槛
    min_len = 2

    # what / why / how 要求更详细一点
    if key in ("what", "why", "how"):
        min_len = 9

    return len(text) >= min_len


def check_topic_completed(slots: dict) -> bool:
    """判断话题是否“足够完整”"""
    must_keys = ["who", "what", "when", "why"]
    optional_keys = ["where", "how"]

    if not all(is_filled_val(slots.get(k), key=k) for k in must_keys):
        return False

    opt_count = sum(
        1 for k in optional_keys
        if is_filled_val(slots.get(k), key=k)
    )
    return opt_count >= 1


# ========= 3. 对话助手 =========
DIALOGUE_AGENT_SYSTEM = """
你是一个中文对话助手，目标是和用户自然地聊天、共情，同时在合适的时候引导用户把一个话题讲完整。

系统会给你当前的“话题类型”和已经掌握的六个要素（who / what / where / when / why / how）。

请注意以下规则：

1. 首先真诚地回应用户的内容：可以共情、安慰、解释、分析、给建议；
2. 然后，根据哪些要素还不清楚，顺带自然地追问一两个问题。
   - 如果某个要素是 null 或者非常笼统/很短（例如 what = “去国外玩了一圈”、why = “就是觉得想出去走走”），
     请把它当成“信息还没讲完”，用自然的问题引导用户具体化：
       * 对 what：可以问“这一路上有没有什么特别难忘的事情？”、“旅途中发生过什么印象深刻的细节吗？”；
       * 对 why：可以问“当时是什么契机让你决定去旅行的？”、“背后有没有什么特别的原因或心情？”；
       * 对 how：可以问“你们是怎么决定具体去哪几个地方的？”、“当时做攻略、订行程是怎么安排的？”；
3. 不要一次性把六个问题全部问完，要根据对话进展慢慢问；
4. 不要自己代替用户下结论或编细节（例如不要帮用户随便猜“可能是为了体验不同文化”），
   这些原因和过程应该由用户自己说出来，你只负责引导；
5. 语气自然、口语化，让用户感觉是在正常聊天，而不是被审问或做问卷调查。

你的输出就是直接发给用户的回复，不要输出 JSON。
"""

def generate_dialogue_reply(history, topic_info: dict, slots: dict) -> str:
    topic_label = topic_info.get("topic_label", "未知话题")
    known_text = []
    for k, name in [
        ("who", "谁（who）"),
        ("what", "发生了什么（what）"),
        ("where", "在哪里（where）"),
        ("when", "什么时候（when）"),
        ("why", "为什么（why）"),
        ("how", "怎么做的 / 打算怎么做（how）"),
    ]:
        v = slots.get(k)
        v_str = v if (isinstance(v, str) and v.strip()) else "未知"
        known_text.append(f"{name}: {v_str}")

    slots_summary = "\n".join(known_text)

    convo_text = ""
    for msg in history:
        role = "用户" if msg["role"] == "user" else "助手"
        convo_text += f"{role}: {msg['content']}\n"

    user_input = f"""
当前话题类型：{topic_label}

当前已知要素为：
{slots_summary}

以下是最近的对话（从早到晚）：
{convo_text}

请你根据上述信息，继续用 1 段话回复用户。
记得先回应用户刚刚说的内容，再自然地引导补充缺失的信息（如果有必要）。
"""

    reply = call_chat_llm(DIALOGUE_AGENT_SYSTEM, user_input)
    return reply


# ========= 4. 情绪分类 =========
EMOTION_CLASSIFIER_SYSTEM = """
你是一个对话情绪分析器。

我会给你一整段用户与助手的对话记录，请你判断“用户在这个话题中整体主导的情绪是什么”。

可选标签包括：
1. 高兴 / 满意
2. 难过 / 沮丧
3. 生气 / 愤怒
4. 紧张 / 焦虑
5. 害怕 / 担心
6. 平静 / 中性

请只输出 JSON，例如：

{
  "label_id": 3,
  "label": "生气 / 愤怒",
  "explanation": "..."
}

分析时请尽量关注用户的语气、用词、评价，而不是助手的内容。
"""

def classify_emotion(history) -> dict:
    convo_text = ""
    for msg in history:
        role = "用户" if msg["role"] == "user" else "助手"
        convo_text += f"{role}: {msg['content']}\n"

    fallback = {
        "label_id": 6,
        "label": "平静 / 中性",
        "explanation": "整体语气比较平稳，没有明显强烈情绪"
    }

    raw = call_chat_llm(EMOTION_CLASSIFIER_SYSTEM, convo_text)
    data = safe_parse_json(raw, fallback)
    if "label" not in data:
        return fallback
    return data


# ========= Streamlit UI =========

st.set_page_config(page_title="话题引导系统Demo", page_icon="💬")
st.title("💬 话题引导系统Demo")

# 初始化会话状态
if "history" not in st.session_state:
    st.session_state.history = []
if "topic_info" not in st.session_state:
    st.session_state.topic_info = None
if "topic_confirmed" not in st.session_state:
    st.session_state.topic_confirmed = False
if "slots" not in st.session_state:
    st.session_state.slots = {k: None for k in ["who", "what", "where", "when", "why", "how"]}
if "completed" not in st.session_state:
    st.session_state.completed = False
if "emotion" not in st.session_state:
    st.session_state.emotion = None

# 👉 新增：话题分类失败次数 & 是否进入“手动选择模式”
if "topic_retry_count" not in st.session_state:
    st.session_state.topic_retry_count = 0
if "manual_topic_select" not in st.session_state:
    st.session_state.manual_topic_select = False

if "rejected_topics" not in st.session_state:
    st.session_state.rejected_topics = []  # 存 topic_id 列表

# ====== 侧边栏：状态与控制 ======
with st.sidebar:
    st.header("会话状态")

    if st.button("🔄 重置会话"):
        st.session_state.history = []
        st.session_state.topic_info = None
        st.session_state.topic_confirmed = False
        st.session_state.slots = {k: None for k in ["who", "what", "where", "when", "why", "how"]}
        st.session_state.completed = False
        st.session_state.emotion = None
        st.session_state.topic_retry_count = 0
        st.session_state.manual_topic_select = False
        st.session_state.rejected_topics = []
        st.rerun()

    # ========== 话题类型展示 & 操作 ==========
    if st.session_state.manual_topic_select and st.session_state.topic_info is None:
        # 已经连续三次判断被否定，交给用户自己选
        st.markdown("### 话题类型：手动选择")
        st.warning("我已经尝试判断了几次，可能还是对不上，这次交给你来选一个最贴近的类型～")

        options = [t["label"] for t in TOPIC_TYPES]
        # 用 key 确保不会每次重置选择
        choice = st.selectbox(
            "请选择这次聊天最接近的话题类型：",
            options,
            key="manual_topic_choice"
        )

        if st.button("✅ 使用这个话题类型"):
            # 由用户手动选择
            for t in TOPIC_TYPES:
                if t["label"] == choice:
                    st.session_state.topic_info = {
                        "topic_id": t["id"],
                        "topic_label": t["label"],
                        "reason": "由用户手动选择话题类型"
                    }
                    break
            st.session_state.topic_confirmed = True
            st.session_state.manual_topic_select = False
            st.session_state.topic_retry_count = 0
            st.rerun()

    else:
        # 正常的自动分类流程
        if st.session_state.topic_info is None:
            st.write("话题类型：尚未识别（请输入第一句话）")
        else:
            st.write(f"识别的话题类型：**{st.session_state.topic_info['topic_label']}**")
            st.caption(f"模型理由：{st.session_state.topic_info.get('reason', '')}")

            if not st.session_state.topic_confirmed:
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ 这个差不多"):
                        st.session_state.topic_confirmed = True
                        st.rerun()
                with col2:
                    if st.button("❌ 不太对，换一个"):
                        # ① 把当前判断的 topic_id 记入“已拒绝列表”
                        curr = st.session_state.topic_info
                        if curr is not None:
                            tid = curr.get("topic_id")
                            if isinstance(tid, int):
                                if tid not in st.session_state.rejected_topics:
                                    st.session_state.rejected_topics.append(tid)
                        # 用户否认本次判断
                        st.session_state.topic_retry_count += 1

                        if st.session_state.topic_retry_count >= 3:
                            # 连续三次被否认：进入手动选择模式
                            st.session_state.manual_topic_select = True
                            st.session_state.topic_info = None
                            st.session_state.history.append({
                                "role": "assistant",
                                "content": (
                                    "看来我连续几次都没猜准你想聊的话题类型，这次就不勉强了～\n\n"
                                    "请你在左侧侧边栏自己选一个最贴近的类型，我们再继续往下聊。"
                                )
                            })
                            st.rerun()
                        else:
                            # < 3 次：请用户多提供一点细节，再重新自动判断
                            st.session_state.topic_info = None
                            st.session_state.history.append({
                                "role": "assistant",
                                "content": (
                                    "好，我明白了，可能是我刚才理解得还不够准确。\n\n"
                                    "你可以再多讲一点这个话题，比如和谁有关、发生在什么时候、"
                                    "对你来说最重要的部分是什么，我会根据你补充的内容重新判断一次类型。"
                                )
                            })
                            st.rerun()
            else:
                st.success("话题类型已确认")

    # ========== 进度条展示 ==========
    st.markdown("---")
    st.subheader("话题完整度（六要素）")

    slots = st.session_state.slots
    filled_count = sum(
        1 for k, v in slots.items()
        if is_filled_val(v, key=k)
    )

    st.progress(filled_count / 6.0 if 6 else 0.0)
    st.write(f"已填充要素：**{filled_count} / 6**")

    for key, name in [
        ("who", "谁（who）"),
        ("what", "发生了什么（what）"),
        ("where", "在哪里（where）"),
        ("when", "什么时候（when）"),
        ("why", "为什么（why）"),
        ("how", "怎么做的 / 打算怎么做（how）"),
    ]:
        v = slots.get(key)
        icon = "✅" if is_filled_val(v, key=key) else "⬜️"
        text = v if (isinstance(v, str) and v.strip()) else "暂无"
        st.write(f"{icon} **{name}**：{text}")

    # ========== 情绪标签展示 ==========
    st.markdown("---")
    st.subheader("对话情绪（完成后生成）")
    if st.session_state.emotion is None:
        st.write("当前话题尚未完整，暂不分析情绪。")
    else:
        emo = st.session_state.emotion
        st.write(f"主要情绪：**{emo['label']}**")
        st.caption(f"理由：{emo.get('explanation', '')}")



# ====== 主区域：对话展示 ======
for msg in st.session_state.history:
    with st.chat_message("user" if msg["role"] == "user" else "assistant"):
        st.markdown(msg["content"])

# ====== 处理用户输入 ======
user_input = st.chat_input("可以用文字先简单说说你想聊什么")

def process_user_message(text: str):
    # 记录用户发言
    st.session_state.history.append({"role": "user", "content": text})

    # 0）如果已经进入“手动选择话题类型模式”，并且还没选完，就先引导用户去侧边栏选择
    if st.session_state.manual_topic_select and st.session_state.topic_info is None:
        reply = (
            "我这边暂时不再自动判断话题类型啦～\n\n"
            "麻烦你先在左侧侧边栏，从下拉框里选一个最贴近这次想聊的类型，"
            "选好之后我们就按那个方向继续聊下去。"
        )
        st.session_state.history.append({"role": "assistant", "content": reply})
        return

    # 1）如果还没有话题类型（自动分类阶段）
    if st.session_state.topic_info is None:
        topic_info = classify_topic(
            text,
            rejected_topic_ids=st.session_state.rejected_topics
        )
        st.session_state.topic_info = topic_info

        reply = (
            f"我先帮你粗略看了一下，感觉你现在聊的是 **「{topic_info['topic_label']}」** 相关的话题。\n\n"
            "如果你觉得差不多，可以在左侧点击“✅ 这个差不多”。\n"
            "如果觉得不太对，可以点“❌ 不太对，换一个”，"
            "再多跟我说一点细节，我会重新帮你判断；\n"
            "如果连续几次都对不上，就交给你自己在侧边栏选择类型～"
        )
        st.session_state.history.append({"role": "assistant", "content": reply})
        return

    # 2）已经识别，但用户还没在侧边栏确认：先提醒确认
    if not st.session_state.topic_confirmed:
        reply = (
            "我已经根据你的描述识别出了一个话题类型。\n\n"
            "👉 请先在左侧侧边栏选择：是点“✅ 这个差不多”，"
            "还是点“❌ 不太对，换一个”。\n"
            "确认之后，我会按这个方向，慢慢帮你把事情的来龙去脉都理清。"
        )
        st.session_state.history.append({"role": "assistant", "content": reply})
        return

    # 3）话题已经完整且情绪分析也做了：提示可以重开
    if st.session_state.completed and st.session_state.emotion is not None:
        emo = st.session_state.emotion
        reply = (
            f"这个话题我们之前已经聊得比较完整了，主要情绪是「{emo['label']}」。\n\n"
            "如果你想开始一个全新的话题，可以点击左侧“🔄 重置会话”，我们重新来一次。"
        )
        st.session_state.history.append({"role": "assistant", "content": reply})
        return

    # 4）正常推进话题：更新槽位 → 判断是否完整 → 回复 or 情绪分析
    new_slots = extract_slots(st.session_state.history, st.session_state.slots)
    st.session_state.slots = new_slots

    completed_now = check_topic_completed(new_slots)
    st.session_state.completed = completed_now

    if not completed_now:
        reply = generate_dialogue_reply(
            history=st.session_state.history,
            topic_info=st.session_state.topic_info,
            slots=new_slots,
        )
        st.session_state.history.append({"role": "assistant", "content": reply})
    else:
        emotion = classify_emotion(st.session_state.history)
        st.session_state.emotion = emotion

        reply = (
            "谢谢你把这件事从头到尾讲清楚，我大概拼出了一个比较完整的故事。\n\n"
            f"从整段对话里看，你现在主要的情绪是：**{emotion['label']}**。\n"
            f"我的理解是：{emotion.get('explanation', '')}\n\n"
            "如果你愿意，我们也可以在这个情绪的基础上，继续聊聊你接下来想怎么应对；\n"
            "如果想换一个全新的话题，可以在左侧点“🔄 重置会话”。"
        )
        st.session_state.history.append({"role": "assistant", "content": reply})



if user_input:
    process_user_message(user_input)
    st.rerun()
