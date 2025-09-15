# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# 文件名: 3w_chat.py
#
# 作者: 今日搬了什么砖
#
# 描述: 这是一个教学示例，展示如何构建一个多轮对话系统，集成用户画像、情境分析、行动策略规划和多模态处理等功能。
#
# -----------------------------------------------------------------------------


import os
import json
import logging
import textwrap
import re
from enum import Enum
import requests
from typing import List, Dict, Any, Tuple, Iterator, Optional, TypedDict, Annotated
from abc import ABC, abstractmethod
import gradio as gr
from openai import OpenAI, OpenAIError
from pydantic import BaseModel, Field, ValidationError
import base64
import mimetypes
import uuid
from langgraph.graph import StateGraph, END

from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool, BaseTool
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_core.messages import ToolMessage, AIMessage, HumanMessage, BaseMessage, FunctionMessage, SystemMessage, ChatMessage
from copy import deepcopy
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
# ==============================================================================
# 配置 (Configuration)
# ==============================================================================
API_KEY = os.getenv("API_KEY", "YOUR_API_KEY")
BASE_URL = os.getenv(
    "BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

ANALYSIS_MODEL = os.getenv("ANALYSIS_MODEL", "qwen-plus-latest")
STRATEGY_MODEL = os.getenv("STRATEGY_MODEL", "qwen-plus-latest")
RESPONSE_MODEL = os.getenv("RESPONSE_MODEL", "qwen-plus-latest")
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "qwen-plus-latest")
RETRIEVAL_MODEL = os.getenv("RETRIEVAL_MODEL", "qwen-plus-latest")
PROFILE_MODEL = os.getenv("PROFILE_MODEL", "qwen-plus-latest")
VISION_MODEL = os.getenv("VISION_MODEL", "qwen-vl-plus")

HISTORY_COMPRESSION_THRESHOLD = int(
    os.getenv("HISTORY_COMPRESSION_THRESHOLD", 20))
MESSAGES_TO_KEEP_UNCOMPRESSED = int(
    os.getenv("MESSAGES_TO_KEEP_UNCOMPRESSED", 6))
PROFILE_UPDATE_INTERVAL = int(os.getenv("PROFILE_UPDATE_INTERVAL", 1))

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

if not API_KEY or "YOUR_DASHSCOPE_API_KEY" in API_KEY:
    logging.warning("API_KEY 环境变量未设置或无效。请确保API_KEY已正确配置。")

# ==============================================================================
# 数据模型 (Data Models)
# ==============================================================================


class CoreDemand(BaseModel):
    """
    核心需求模型，用于封装从用户输入中提炼出的关键信息。
    """
    key_entities: List[str] = Field(
        default_factory=list,
        description="从用户输入中提取的核心实体或关键词列表。"
    )
    main_question: str = Field(
        default="N/A",
        description="用户输入的核心问题或主要意图。"
    )


class Location(BaseModel):
    """
    地理位置信息。
    """
    city: str = Field(
        default="Unknown",
        description="用户所在的城市。"
    )
    country: str = Field(
        default="Unknown",
        description="用户所在的国家。"
    )


class LanguageProfile(BaseModel):
    """
    用户的语言能力概况。
    """
    primaryLanguage: str = Field(
        default="Unknown",
        description="用户的主要使用语言。"
    )
    spokenLanguages: List[str] = Field(
        default_factory=list,
        description="用户可能会说的其他语言列表。"
    )


class InferredIdentity(BaseModel):
    """
    根据用户信息推断出的身份概况。
    """
    fullName: str = Field(
        default="Unknown",
        description="推断出的用户全名。"
    )
    location: Location = Field(
        default_factory=Location,
        description="推断出的用户地理位置信息。"
    )
    languageProfile: LanguageProfile = Field(
        default_factory=LanguageProfile,
        description="推断出的用户语言能力概况。"
    )


class InferredBigFiveKeywords(BaseModel):
    """
    基于大五人格模型（OCEAN）推断出的相关关键词。
    """
    openness: List[str] = Field(
        default_factory=list,
        description="与'开放性'人格特质相关的关键词列表。"
    )
    conscientiousness: List[str] = Field(
        default_factory=list,
        description="与'尽责性'人格特质相关的关键词列表。"
    )
    extraversion: List[str] = Field(
        default_factory=list,
        description="与'外向性'人格特质相关的关键词列表。"
    )
    agreeableness: List[str] = Field(
        default_factory=list,
        description="与'宜人性'人格特质相关的关键词列表。"
    )
    neuroticism: List[str] = Field(
        default_factory=list,
        description="与'神经质'（或情绪不稳定性）人格特质相关的关键词列表。"
    )


class PersonalityAndValues(BaseModel):
    """
    用户的个性和价值观画像。
    """
    characterTags: List[str] = Field(
        default_factory=list,
        description="描述用户性格特点的标签列表。"
    )
    inferredBigFiveKeywords: InferredBigFiveKeywords = Field(
        default_factory=InferredBigFiveKeywords,
        description="根据大五人格模型推断出的关键词集合。"
    )
    values: List[str] = Field(
        default_factory=list,
        description="用户的价值观或重要信念列表。"
    )


class SentimentPolarity(BaseModel):
    """
    情感极性分析结果。
    """
    averageSentiment: float = Field(
        default=0.5,
        description="平均情感倾向得分，通常在0到1之间（0=负面, 0.5=中性, 1=正面）。"
    )
    volatility: str = Field(
        default="Medium",
        description="情感波动的程度（如：'Low', 'Medium', 'High'）。"
    )


class CommunicationProfile(BaseModel):
    """
    用户的沟通画像，分析其沟通方式和语言特征。
    """
    communicationStyle: List[str] = Field(
        default_factory=list,
        description="描述用户沟通风格的标签列表（例如：'正式', '口语化', '直接', '委婉'）。"
    )
    linguisticMarkers: List[str] = Field(
        default_factory=list,
        description="在沟通中识别出的特定语言特征或口头禅（例如：'嗯', '其实', '总之'）。"
    )
    sentimentPolarity: SentimentPolarity = Field(
        default_factory=SentimentPolarity,
        description="沟通中体现出的整体情感倾向分析。"
    )


class Skill(BaseModel):
    """
    描述单项技能及其熟练程度。
    """
    skillName: str = Field(
        description="具体的技能或能力的名称（例如：'Python编程', '公开演讲'）。"
    )
    proficiency: str = Field(
        description="对该技能的掌握程度（例如：'初学者', '中级', '专家'）。"
    )


class Like(BaseModel):
    """
    描述用户喜欢的某个具体事物及其分类。
    """
    item: str = Field(
        description="用户喜欢的具体事物名称（例如：'咖啡', '爵士乐'）。"
    )
    category: str = Field(
        description="该事物所属的分类（例如：'饮品', '音乐流派'）。"
    )


class Dislike(BaseModel):
    """
    描述用户不喜欢的某个具体事物及其分类。
    """
    item: str = Field(
        description="用户不喜欢的具体事物名称。"
    )
    category: str = Field(
        description="该事物所属的分类。"
    )


class KnowledgeAndInterests(BaseModel):
    """
    用户的知识领域和兴趣爱好。
    """
    topics: List[str] = Field(
        default_factory=list,
        description="用户感兴趣或具备知识的主题领域列表（例如：'人工智能', '古代史'）。"
    )
    skills: List[Skill] = Field(
        default_factory=list,
        description="用户掌握的技能及其熟练度列表。"
    )
    hobbies: List[str] = Field(
        default_factory=list,
        description="用户的业余爱好或兴趣列表（例如：'徒步', '绘画'）。"
    )
    likes: List[Like] = Field(
        default_factory=list,
        description="用户明确表示喜欢的事物列表。"
    )
    dislikes: List[Dislike] = Field(
        default_factory=list,
        description="用户明确表示不喜欢的事物列表。"
    )


class MentionedGoal(BaseModel):
    """
    描述一个用户提及的具体目标。
    """
    title: str = Field(
        description="目标的具体名称或简短描述（例如：'学习西班牙语'，'完成一个个人项目'）。"
    )
    category: str = Field(
        description="该目标所属的类别（例如：'职业发展', '个人成长', '健康'）。"
    )
    status: str = Field(
        description="目标的当前进展状态（例如：'计划中', '进行中', '已完成'）。"
    )


class GoalsAndAspirations(BaseModel):
    """
    用户的目标与抱负的集合。
    """
    mentionedGoals: List[MentionedGoal] = Field(
        default_factory=list,
        description="用户在沟通过程中提及的目标列表。"
    )


class UserProfile(BaseModel):
    inferredIdentity: InferredIdentity = Field(
        default_factory=InferredIdentity)
    personalityAndValues: PersonalityAndValues = Field(
        default_factory=PersonalityAndValues)
    communicationProfile: CommunicationProfile = Field(
        default_factory=CommunicationProfile)
    knowledgeAndInterests: KnowledgeAndInterests = Field(
        default_factory=KnowledgeAndInterests)
    goalsAndAspirations: GoalsAndAspirations = Field(
        default_factory=GoalsAndAspirations)


class TopicAnalysis(BaseModel):
    current_topic: str = "N/A"
    is_new_topic: bool = True
    continuity_reasoning: str = "Initial message."


class Confidence(str, Enum):
    HIGH = "高"
    MEDIUM = "中"
    LOW = "低"


class ImplicitIntentGuess(BaseModel):
    guess: str = Field(..., description="对用户潜在意图的具体猜测。")
    confidence: Confidence = Field(..., description="对此猜测的置信度，分为高、中、低三档。")


class IntentAnalysis(BaseModel):
    explicit_intent: str = "N/A"
    implicit_intents: List[ImplicitIntentGuess] = Field(
        default_factory=list,
        description="对用户潜在意图的猜测列表（最多3个），按置信度降序排列。"
    )
    sentiment: str = "Neutral"


class SituationalSnapshot(BaseModel):
    who_profile: UserProfile
    what_topic: TopicAnalysis
    why_intent: IntentAnalysis
    how_demand: CoreDemand


class ActionStrategy(Enum):
    DIRECT_ANSWER = "直接解答"
    CLARIFY_ASK = "澄清反问"
    PROACTIVE_GUIDE = "主动引导"
    EXECUTE_TASK = "执行任务"
    RAG_QUERY = "检索增强生成"
    TOOL_USE = "工具调用"
    SEARCH_AGENT_DELEGATION = "搜索代理"


class ToolCallRequest(BaseModel):
    """表示单个工具调用的请求。"""
    name: str = Field(..., description="要调用的工具的名称。")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="工具所需的参数。")


class ActionPlan(BaseModel):
    strategy: ActionStrategy
    reasoning: str
    proposed_response_outline: str
    clarification_question: str = ""
    search_query: str = ""
    tool_calls: List[ToolCallRequest] = Field(
        default_factory=list, description="当策略为 '工具调用' 时，要执行的一个或多个工具调用列表。")


class SituationalAnalysisOutput(BaseModel):
    how_demand: CoreDemand = Field(default_factory=CoreDemand)
    what_topic_analysis: TopicAnalysis = Field(default_factory=TopicAnalysis)
    why_intent_analysis: IntentAnalysis = Field(default_factory=IntentAnalysis)


class Persona(BaseModel):
    name: str = Field(description="人格的唯一名称，用于UI显示。")
    description: str = Field(description="给LLM的详细指令，描述了这个角色的行为、说话风格、背景等。")


class GraphState(TypedDict):
    conversation_history: List[Dict[str, Any]]
    user_profile: UserProfile
    user_input: str
    snapshot: Optional[SituationalSnapshot]
    plan: Optional[ActionPlan]
    rag_context: Optional[str]
    final_response: Optional[str]
    turn_count: int
    active_persona: Persona
    image_b64_urls: Optional[List[str]]


class PersonaManager:

    def __init__(self):
        self._personas: Dict[str, Persona] = {}
        self._default_persona_name: Optional[str] = None
        self._load_default_personas()

    def _load_default_personas(self):
        default_personas = [
            Persona(
                name="专业AI助手",
                description="你的角色是一个专业、严谨、客观的AI助手。你的回答应该结构清晰、逻辑性强、内容准确。避免使用口语化、情绪化的表达。优先使用正式和书面的语言风格。"
            ),
            Persona(
                name="友好伙伴",
                description="你的角色是一个热情、友好、乐于助人的伙伴。你的回答应该充满鼓励和积极性。可以使用一些轻松的语气词和表情符号（如😊、👍）。多使用“我们”来拉近与用户的距离。"
            ),
            Persona(
                name="创意缪斯",
                description="你的角色是一个充满想象力和创造力的灵感激发者。你的回答应该不拘一格、富有创意，并能从独特的角度提出问题和看法。多使用比喻和生动的描述。鼓励用户进行头脑风暴。"
            ),
            Persona(
                name="易经高手",
                description="你的角色是一个精通《周易》的智慧导师。回答时应结合卦象、阴阳五行、变易之理进行解释。语言风格庄重而含蓄，善用比喻和象征，启发用户思考人生与选择。"
            ),
            Persona(
                name="天真小孩",
                description="你的角色是一位天真、好奇的小朋友。回答应该简单、直白、充满童趣。可以多用‘为什么呀？’‘好玩！’等语气，表达纯真和好奇心。"
            ),
            Persona(
                name="玄学大师",
                description="你的角色是一位神秘、深邃的玄学大师。回答应当带有哲理与象征，融合星象、风水、命理等元素，语气悠远飘渺，仿佛在揭示隐藏的真理。"
            ),
            Persona(
                name="股市老手",
                description="你的角色是一位经验丰富的股市投资老手。回答应该结合行情走势、投资逻辑与市场心理。风格务实、直白，带有一些老江湖的口吻，比如‘市场永远是对的’。"
            ),
            Persona(
                name="冷酷逻辑学家",
                description="你的角色是一位极度理性、冷静的逻辑学家。回答必须严格遵循推理与演绎，避免感性表达。语言风格简洁、冷峻，强调前提、结论与逻辑一致性。"
            ),
            Persona(
                name="励志教练",
                description="你的角色是一位充满激情的励志教练。回答应该振奋人心，充满行动力和正能量。常用激励性的词语，如‘你一定可以做到！’‘突破极限！’。"
            ),
            Persona(
                name="文艺诗人",
                description="你的角色是一位文艺气质的诗人。回答应该带有诗意与情感，用隐喻、意象和优美的语言描绘想法。回答可以像散文或诗歌，触动用户的情绪。"
            ),
            Persona(
                name="冷面哲学家",
                description="你的角色是一位洞察世事的哲学家。回答应充满思辨和反问，引导用户深度思考人生意义与价值。语言风格冷静而犀利，直指本质。"
            ),
            Persona(
                name="互联网段子手",
                description="你的角色是一位擅长用幽默与梗交流的网友。回答应该轻松搞笑，带点调侃与机智。可以使用网络流行语和表情包风格，让氛围更活跃。"
            ),
            Persona(
                name="严厉导师",
                description="你的角色是一位要求严格、直言不讳的导师。回答应当直截了当，不粉饰太平，指出问题并提出改进方向。语气犀利但出于关心。"
            ),
            Persona(
                name="未来先知",
                description="你的角色是一位仿佛来自未来的先知。回答应神秘而前瞻，充满预言和未来视角。常用‘在未来，你会发现……’之类的表述。"
            )
        ]

        for p in default_personas:
            self.add_persona(p)

        if default_personas:
            self._default_persona_name = default_personas[0].name

    def add_persona(self, persona: Persona):
        self._personas[persona.name] = persona

    def get_persona(self, name: str) -> Optional[Persona]:
        return self._personas.get(name)

    def get_default_persona(self) -> Optional[Persona]:
        if self._default_persona_name:
            return self.get_persona(self._default_persona_name)
        return None

    def list_persona_names(self) -> List[str]:
        return list(self._personas.keys())


# ==============================================================================
# 服务层 (Services & Logic)
# ==============================================================================


class AIClientBase:
    def get_chat_completion(self, messages: List[Dict[str, Any]], model: str, temperature: float, is_json: bool = False, urls: Optional[List] = None) -> Dict[str, Any]:
        raise NotImplementedError

    def stream_chat_completion(self, messages: List[Dict[str, Any]], model: str, temperature: float) -> Iterator[str]:
        raise NotImplementedError

    def describe_image(self, base64_data_urls: List[str], prompt: str, model: str) -> str:
        raise NotImplementedError


class OpenAIClient(AIClientBase):
    def __init__(self, api_key: str, base_url: str):
        if not api_key or "YOUR_DASHSCOPE_API_KEY" in api_key:
            raise ValueError("API key not set or is a placeholder.")
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def get_chat_completion(self, messages: List[Dict[str, Any]], model: str, temperature: float, is_json: bool = False, tools: Optional[List] = None) -> Dict[str, Any]:
        try:
            kwargs = {"model": model, "messages": messages,
                      "temperature": temperature}
            if is_json:
                kwargs["response_format"] = {"type": "json_object"}
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.model_dump()
        except OpenAIError as e:
            logging.error(f"OpenAI API 错误: {e}")
            return {"role": "assistant", "content": json.dumps({"error": f"API Error: {e}"})}

    # <--- 修改签名
    def describe_image(self, base64_data_urls: List[str], prompt: str, model: str) -> str:
        """使用多模态模型描述一个或多个图片的内容。"""
        if not base64_data_urls:
            return "处理图片时出错：未提供图片数据。"

        # 构建多模态消息内容
        content = [{"type": "text", "text": prompt}]
        for data_url in base64_data_urls:
            content.append({
                "type": "image_url",
                "image_url": {"url": data_url},
            })

        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
                max_tokens=4096,  # 增加token以容纳多张图片的描述
            )
            description = response.choices[0].message.content
            return description or "无法从图片中提取描述。"
        except OpenAIError as e:
            logging.error(f"多模态 API 错误: {e}")
            return f"调用多模态模型时出错: {e}"

    def stream_chat_completion(self, messages: List[Dict[str, Any]], model: str, temperature: float) -> Iterator[str]:
        try:
            stream = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                stream=True
            )
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
        except OpenAIError as e:
            logging.error(f"OpenAI API 流式错误: {e}")
            yield f"抱歉，流式响应出错: {e}"


class GoogleClient(AIClientBase):
    def __init__(self, api_key: str, base_url: str):
        self.llm = GoogleGenerativeAI(
            model="qwen-plus-latest", google_api_key=api_key)

    def get_chat_completion(self, messages: List[Dict[str, Any]], model: str, temperature: float, is_json: bool = False, tools: Optional[List] = None) -> Dict[str, Any]:
        raise NotImplementedError


def create_llm_client(privider: str) -> AIClientBase:
    if privider.lower() == "openai":
        return OpenAIClient(API_KEY, BASE_URL)
    elif privider.lower() == "google":
        return GoogleClient(API_KEY, BASE_URL)
    else:
        raise ValueError(f"Unsupported provider: {privider}")


class HistoryManager:
    def __init__(self, client: OpenAIClient):
        self.client = client
        self.compression_prompt = textwrap.dedent("""
            你是一个对话摘要AI。请将以下对话历史浓缩成一段简洁的摘要，保留关键信息、实体、用户核心意图和已达成的结论。
            这段摘要将作为未来对话的上下文记忆。请直接输出摘要内容，不要添加任何额外的前言或结语。
        """)

    def get_text_from_message(self, message):
        if isinstance(message.content, str):
            return message.content
        elif isinstance(message.content, list):
            return " ".join(
                part["content"] for part in message.content if part["type"] == "text"
            )
        return ""

    def format_history_for_model(self, history: List[BaseMessage]) -> str:
        return "\n".join(f"{msg.type}: {self.get_text_from_message(msg)}" for msg in history)

    def compress_history_if_needed(self, history: List[BaseMessage]) -> List[BaseMessage]:
        if len(history) <= HISTORY_COMPRESSION_THRESHOLD:
            return history
        logging.info(
            f"对话历史长度 ({len(history)}) 已超过阈值 ({HISTORY_COMPRESSION_THRESHOLD})，开始压缩。")
        to_compress = history[:-MESSAGES_TO_KEEP_UNCOMPRESSED]
        to_keep = history[-MESSAGES_TO_KEEP_UNCOMPRESSED:]

        history_str = self.format_history_for_model(to_compress)

        messages = [{"role": "system", "content": self.compression_prompt}, {
            "role": "user", "content": history_str}]
        summary_response = self.client.get_chat_completion(
            messages, SUMMARY_MODEL, 0.2)
        summary_content = summary_response.get('content')

        if "error" in summary_response or not summary_content:
            logging.error("历史摘要生成失败，将保留原始历史。")
            return history

        new_history = [ToolMessage(
            name="summary",
            tool_call_id=str(uuid.uuid4()),
            content=f"前情提要（对话摘要）:\n{summary_content}")] + to_keep
        logging.info("历史压缩完成。")
        return new_history


class SessionManager:
    def __init__(self):
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._session_counter = 0

    def create_session(self) -> Tuple[str, str]:
        self._session_counter += 1
        session_id = str(uuid.uuid4())
        display_name = f"会话 {self._session_counter}"
        self._sessions[session_id] = {
            "display_name": display_name,
            "conversation_history": [],
            "user_profile": UserProfile(),
            "turn_count": 0,
            "active_persona_name": "专业AI助手"
        }
        logging.info(f"创建新会话: {display_name} (ID: {session_id})")
        return session_id, display_name

    def update_persona_for_session(self, session_id: str, persona_name: str):
        if session_id in self._sessions:
            self._sessions[session_id]["active_persona_name"] = persona_name
            logging.info(f"会话 {session_id} 的人格已更新为: {persona_name}")

    def delete_session(self, session_id: str):
        if session_id in self._sessions:
            display_name = self._sessions[session_id]["display_name"]
            del self._sessions[session_id]
            logging.info(f"删除会话: {display_name} (ID: {session_id})")

    def get_session_list(self) -> List[Tuple[str, str]]:
        return sorted(
            [(data["display_name"], session_id)
             for session_id, data in self._sessions.items()],
            key=lambda x: int(re.search(r'\d+', x[0]).group())
        )

    def get_session_data(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self._sessions.get(session_id)

    def update_session_data(self, session_id: str, final_graph_state: Dict[str, Any]):
        if session_id in self._sessions:
            session = self._sessions[session_id]
            session["turn_count"] += 1
            if 'conversation_history' in final_graph_state:
                session["conversation_history"] = final_graph_state['conversation_history']
            if 'user_profile' in final_graph_state:
                session["user_profile"] = final_graph_state['user_profile']
            logging.info(f"更新会话 '{session['display_name']}' 的状态。")


MESSAGE_TYPE = {
    "ai": "assistant",
    "human": "user",
    "tool": "tool",
    "system": "system",
    "function": "function",
    "chat": "system"
}


class PromptFactory:
    """提示词工厂，为AI的每个思考步骤构建专用提示。"""

    def get_text_from_content(self, content):
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            return " ".join(
                part["content"] for part in content if part["type"] == "text"
            )
        return ""

    def create_situational_analysis_prompt(self, user_input: str, history: List[Dict[str, str]], profile: UserProfile) -> str:
        history_str = "\n".join(
            f"{MESSAGE_TYPE[msg.type]}: {self.get_text_from_content(msg.content)}" for msg in history)
        # 传递 profile 字符串仍然有用，因为它可以为意图和主题分析提供上下文
        profile_str = profile.model_dump_json(indent=2)
        return textwrap.dedent(f"""
        ## role ##
        You are a world-class conversation analysis expert. Your expertise lies in deconstructing a user's latest message to understand their immediate needs, including their unspoken intentions.

        ## task ##
        Your core task is to conduct a focused analysis of the user's latest message (`{{user_input}}`), considering the conversation history (`{{history_str}}`) and user profile (`{{profile_str}}`) for context. Your goal is to understand the user's demand, the current topic, and their intent **in this specific turn**.

        ## Requirements ##
        1.  **Analyze Demand (`how_demand`):** Extract key entities from `{{user_input}}` and summarize the user's core question or instruction into `main_question`.
        2.  **Analyze Topic (`what_topic_analysis`):** Identify the primary topic of `{{user_input}}`. Determine if it's a new topic compared to the history and provide a brief justification.
        3.  **Analyze Intent (`why_intent_analysis`):** 
            - Decipher the user's `explicit_intent`.
            - **Crucially, identify up to three potential `implicit_intents` (the user's unspoken goals or underlying needs). For each guess, assign a confidence level ("高", "中", or "低"). This should be a list of objects.**
            - Classify the `sentiment` of the message.
        4.  **Language Consistency:** All string values in the JSON output must match the language of `{{user_input}}`.
        5.  **Strict JSON Output:** Your entire output must be a single, raw JSON object adhering to the following structure. Pay close attention to the structure of `implicit_intents`.
        6. The language of your response **MUST match the language of the user's input **. (e.g., if the user asks in Chinese, respond in Chinese; if in English, respond in English).
        {{
            "how_demand": {{"key_entities": [], "main_question": "..."}},
            "what_topic_analysis": {{"current_topic": "...", "is_new_topic": boolean, "continuity_reasoning": "..."}},
            "why_intent_analysis": {{
                "explicit_intent": "...", 
                "implicit_intents": [
                    {{"guess": "用户的第一个潜在意图", "confidence": "高"}},
                    {{"guess": "用户的第二个潜在意图", "confidence": "中"}},
                    {{"guess": "用户的第三个潜在意图", "confidence": "低"}}
                ],
                "sentiment": "..."
            }}
        }}

        **Input:**
        `profile_str`: `{profile_str}`
        `history_str`: `{history_str}`
        `user_input`: `{user_input}`
        """)

    def create_profile_update_prompt(self, history: List[Dict[str, str]], profile: UserProfile) -> str:

        history_str = "\n".join(
            f"{MESSAGE_TYPE[msg.type]}: {self.get_text_from_content(msg.content)}" for msg in history)
        profile_str = profile.model_dump_json(indent=2, exclude_unset=True)
        profile_schema_str = UserProfile.model_json_schema()

        return textwrap.dedent(f"""
        # 角色
        你是一个心理画像专家 AI。你的唯一职责是分析对话，构建一个关于用户人格、价值观与目标的动态、持续演进的理解。你的任务是分析传入的 `conversation_history`，对比 `existing_profile`，并生成一个**仅包含新增或更新信息的 JSON Patch**。你的核心价值在于超越表面信息，进行综合推理，并产出可追溯的洞见。

        # 分析流程：严格两步法

        请对每一次分析**严格执行**以下两步流程。

        ## 第一步 —— 证据提取（观察）
        在对话中寻找原始、客观的数据点。此阶段只做信息收集，不做价值判断。
        -   **明确陈述：** 标注用户直接提到的事实信息，如姓名、地点、技能、爱好、好恶等（例如：“我正在学 Python”、“我受不了堵车”）。
        -   **语言特征：** 记录用户的说话方式：正式还是随意？是否使用行业术语？句子长短？倾向于提问还是陈述？
        -   **情绪语气：** 判断对话的整体情绪：积极、消极还是中性？是否存在强烈的情绪波动（如兴奋、沮丧、愤怒）？
        -   **反复出现的主题：** 用户经常谈论哪些话题？什么事情最占据他们的注意力？

        ## 第二步 —— 洞见合成（推断）
        这是你的核心功能。将第一步收集的证据串联成推理链，推断出潜在的心理特质。所有推断都必须有据可循。在填充 `personalityAndValues` 字段时，**优先使用下面的五大人格模型（Big Five）作为你的核心分析框架**。

        ### **聚焦五大人格模型 (Big Five) 的推断指南**

        #### 1. 开放性 (Openness - O)
        -   **核心定义：** 对新思想、艺术、情感、冒险和不寻常体验的开放与接纳程度。
        -   **寻找信号：**
            -   谈论艺术、音乐、文学，或表达对美的欣赏。
            -   表现出强烈的好奇心，喜欢学习新知识或新技能。
            -   喜欢抽象思考、讨论哲学或复杂概念。
            -   乐于尝试新事物、去新的地方或打破常规。
        -   **推断示例：**
            -   用户说：“我喜欢逛博物馆，思考画作背后的象征意义。” -> 推断 `inferredBigFiveKeywords.openness: ["审美", "富有想象力"]`
            -   用户问：“这个技术背后的原理是什么？我想深入了解一下。” -> 推断 `inferredBigFiveKeywords.openness: ["好奇", "智力驱动"]`

        #### 2. 尽责性 (Conscientiousness - C)
        -   **核心定义：** 自我约束、有条理、目标导向以及遵守规范的程度。
        -   **寻找信号：**
            -   谈论计划、日程、截止日期和目标。
            -   表现出对细节的关注和对工作的严谨。
            -   提及自律行为（如坚持锻炼、学习打卡）。
            -   表达对责任感和可靠性的重视。
        -   **推断示例：**
            -   用户说：“我必须在周五前完成这个项目，所以我已经列好了每天的任务清单。” -> 推断 `inferredBigFiveKeywords.conscientiousness: ["有条理", "目标驱动"]`
            -   用户说：“虽然很想放松，但我还是坚持完成了今天的学习任务。” -> 推断 `inferredBigFiveKeywords.conscientiousness: ["自律", "尽责"]`

        #### 3. 外向性 (Extraversion - E)
        -   **核心定义：** 从社交互动中获取能量的程度，以及热情、自信和社交活跃度。
        -   **寻找信号：**
            -   喜欢谈论与朋友、团队的活动，享受成为众人瞩目的焦点。
            -   语言充满能量、热情洋溢。
            -   在对话中表现得健谈、果断、主动。
            -   反向信号（内向）：偏爱独处、安静的环境，谈话更深入而非广泛。
        -   **推断示例：**
            -   用户说：“周末跟一大群朋友出去玩真是太棒了，给我充满了电！” -> 推断 `inferredBigFiveKeywords.extraversion: ["社交活跃", "精力充沛"]`
            -   用户说：“我更喜欢和一两个知心朋友深入聊天，而不是参加大型派对。” -> 推断 `inferredBigFiveKeywords.extraversion: ["内省", "安静"]` (此为低外向性表现)

        #### 4. 宜人性 (Agreeableness - A)
        -   **核心定义：** 在社交中表现出的同情心、合作性、信任和利他倾向。
        -   **寻找信号：**
            -   表达对他人的关心和同理心。
            -   在对话中使用合作性语言（如“我们”、“一起”）。
            -   倾向于避免冲突，寻求和谐。
            -   表现出对他人动机的信任。
        -   **推断示例：**
            -   用户说：“我能理解他的难处，我们应该想办法帮帮他。” -> 推断 `inferredBigFiveKeywords.agreeableness: ["有同理心", "乐于助人"]`
            -   用户说：“我觉得争论这个没意义，大家各退一步吧。” -> 推断 `inferredBigFiveKeywords.agreeableness: ["合作", "寻求和谐"]`

        #### 5. 情绪稳定性 (Neuroticism - N 的反面)
        -   **核心定义：** 情绪的稳定程度。低神经质性（高稳定性）意味着平静、自信、不易焦虑。高神经质性意味着容易体验到焦虑、愤怒、沮丧等负面情绪。
        -   **寻找信号：**
            -   频繁表达担忧、焦虑或压力。
            -   对小事反应过度，情绪波动大。
            -   表现出自我怀疑或悲观的看法。
            -   反向信号（高稳定性）：在谈论挑战时表现出冷静、乐观的态度。
        -   **推断示例：**
            -   用户说：“这点小事又让我焦虑了一整天，总是担心会出岔子。” -> 推断 `inferredBigFiveKeywords.neuroticism: ["易焦虑", "担忧"]`
            -   用户说：“虽然遇到了困难，但我相信总有解决办法的，不急。” -> 推断 `inferredBigFiveKeywords.neuroticism: ["沉着", "情绪稳定"]`

        # 高质量推断的指导原则
        -   **超越字面：** 你的价值在于发现“为什么”，而不仅是“是什么”。优先识别动机、价值观和内在倾向。
        -   **以证据为根基：** 输出的每一条推断都应能明确追溯到对话中的具体证据。
        -   **保守原则：** 如果信号薄弱或模棱两可，**绝不包含**该推断。宁可返回空对象，也不要输出低置信度的信息。
        -   **合成而非罗列：** 关注模式。例如，用户提到了三款不同的设计工具，关键洞见不是列出这三项技能，而是标注 `characterTags: ["创意工作者", "设计师"]`。

        # 绝对指令（必须严格遵守）
        **以下规则是强制性的，必须无条件执行：**
        1.  **语言一致性优先：** 所有输出的字符串值的语言**必须**与 `conversation_history` 中用户使用的语言保持完全一致。
        2.  **只打补丁，不替换 (PATCH, DON'T REPLACE)：** 输出的 JSON **只能**包含新增或发生变化的键值对。
        3.  **纯净 JSON 输出：** 最终输出必须是一个单一、语法完全正确的 JSON 对象。禁止包含任何解释性文字、注释或 Markdown 格式。
        4.  **无变更则返回空对象：** 如果无需任何新增或更新，必须返回一个空的 JSON 对象：`{{}}`。
        5.  **列表更新规则：** 简单列表只包含**新增项**；对象列表包含**完整的新增对象**。

        # 端到端教学示例
        **`existing_profile`（输入）：**
        ```json
        {{
        "knowledgeAndInterests": {{
            "topics": ["科技"]
        }}
        }}
        ```
        **`conversation_history`（输入）：**
        ```
        user: 我最近在学着自己做木工，虽然过程很慢，需要极大的耐心和精确的计划，但看着一块原始的木头在自己手里慢慢变成一件有用的家具，那种成就感真的无可替代。我希望能做出既美观又实用的东西。
        ```
        **正确的 JSON Patch 输出示例：**
        ```json
        {{
        "personalityAndValues": {{
            "inferredBigFiveKeywords": {{
            "conscientiousness": ["有条理", "耐心"],
            "openness": ["审美"]
            }},
            "values": ["成就感", "实用主义", "美学"],
            "characterTags": ["创造者", "手艺人"]
        }},
        "knowledgeAndInterests": {{
            "hobbies": ["木工"]
        }}
        }}
        ```

        ---

        # 模式参考（字段名与结构）
        ```json
        {json.dumps(profile_schema_str, indent=2)}
        ```

        ---

        # 现在开始你的任务
        请应用上述严格流程分析下面数据。

        **`existing_profile`:**
        ```json
        {profile_str}
        ```

        **`conversation_history`:**
        ```
        {history_str}
        ```
    """)

    def create_strategy_prompt(self, snapshot: SituationalSnapshot, history: str, openai_tools: List[Dict]) -> str:
        snapshot_str = snapshot.model_dump_json(indent=2)
        strategies = [s.value for s in ActionStrategy]
        tools_str = json.dumps(openai_tools, indent=2, ensure_ascii=False)
        return textwrap.dedent(f"""
        ## role ##
        你是一个高度专业化、逻辑严谨的AI策略指挥官。你的唯一使命是作为决策核心，分析输入情景，并输出一个规定下一步行动的、单一且格式绝对正确的JSON命令。
        ## task ##
        接收并分析一个包含用户需求、可用工具和策略的输入包。从可用策略中选择唯一最有效的行动方案，并构建一个严格符合下方规范的JSON对象来指令系统执行该行动。你的输出必须是纯粹的JSON，不包含任何额外的解释、对话或格式化字符。

        第一步：分析需求 (Analyze)
        - 深入解读情景快照中`how_demand.main_question`字段。**特别注意`why_intent.implicit_intents`列表，优先考虑置信度为“高”的潜在意图，但也要综合评估其他可能性，以制定最稳健的策略。**
        第二步：选择策略 (Decide)
        - 基于下方的 **策略选择指南**，从可用策略列表中选择一个最能直接、高效地满足用户需求的策略。
        第三步：阐述理由 (Justify)
        - 在最终JSON输出的`reasoning`字段中，用一句话简洁地解释你为什么选择该策略，并说明它为何优于其他选项。
        第四步：构建命令 (Construct)
        - 根据下方输出规范中的严格模式和条件逻辑，生成最终的单一JSON对象。

        ---
        ### **策略选择指南 (Strategy Selection Guide)** ###

        **请严格按照以下标准进行选择：**

        1.  **`{ActionStrategy.RAG_QUERY.value}` (检索增强生成):**
            *   **何时使用:** 当用户的问题很可能可以从一个**特定的、内部的、私有的知识库**中找到答案时使用。这适用于查询事实、数据或已归档的信息。
            *   **思考过程:** “这个问题看起来像是在查阅一份内部文档或数据库。我应该在本地知识库里找找。”
            *   **示例:** "查一下腾讯的简介" (如果知识库中有公司资料)。

        2.  **`{ActionStrategy.SEARCH_AGENT_DELEGATION.value}` (委托搜索代理):**
            *   **何时使用:** 当用户需要**开放域的、公共的、最新的信息**时使用。这适用于定义、概念解释、新闻、教程或任何通常需要用搜索引擎才能找到的内容。
            *   **思考过程:** “这个问题很宽泛，或者需要最新的信息，本地知识库里肯定没有。我需要让一个专门的搜索代理去网上查。”
            *   **示例:** "什么是 LangGraph？", "RAG 和 Agent Memory 有什么区别？"。

        3.  **`{ActionStrategy.TOOL_USE.value}` (工具调用):**
            *   **何时使用:** 当用户的请求可以直接通过调用一个或多个**具体的功能性工具**来完成时使用。这适用于执行动作，而非查找信息。
            *   **思考过程:** “这个任务（比如计算、查财报）有现成的工具可以完美解决。”
            *   **示例:** "计算 100 * (5+3)", "阿里巴巴的收入是多少？"。

        4.  **`{ActionStrategy.DIRECT_ANSWER.value}` (直接解答):**
            *   **何时使用:** 当你可以完全依靠自己的内置知识库，无需任何外部信息或工具就能提供高质量回答时。
            *   **思考过程:** “这个问题很简单，我可以直接回答。”
            *   **示例:** "你好", "写一首关于春天的诗"。
        ---

        **严格的 JSON 输出模式：**
        你**必须**输出一个单一、有效的 JSON 对象，不得包含任何其他内容。

        {{
        "strategy": "string",
        "reasoning": "string",
        "proposed_response_outline": "string",
        "clarification_question": "string",
        "search_query": "string",
        "tool_calls": [
            {{
            "name": "string",
            "parameters": {{}}
            }}
        ]
        }}


        **强制性条件逻辑：**
        *   **如果 `strategy` 是 `"{ActionStrategy.SEARCH_AGENT_DELEGATION.value}"`:**
            *   用一个适合搜索引擎的简洁查询字符串填充 `search_query` 字段。
            *   所有其他特定于行动的字段**必须**为空 (`""` 或 `[]`)。
        *   **如果 `strategy` 是 `"{ActionStrategy.RAG_QUERY.value}"`:**
            *   用一个简洁的搜索查询字符串填充 `search_query` 字段。
            *   所有其他特定于行动的字段**必须**为空 (`""` 或 `[]`)。
        *   **如果 `strategy` 是 `"{ActionStrategy.TOOL_USE.value}"`:**
            *   `tool_calls` 列表必须包含至少一个工具对象。
            *   所有其他字段 (search_query, clarification_question, proposed_response_outline) 必须为空 ("" 或 [])。
        *   **如果 `strategy` 是 `"{ActionStrategy.CLARIFY_ASK.value}"`：**
            *   用一个向用户提出的问题填充 `clarification_question` 字段。
            *   所有其他特定于行动的字段**必须**为空 (`""` 或 `[]`)。
        *   **如果 `strategy` 是任何其他值：**
            *   用建议回复的简要大纲填充 `proposed_response_outline` 字段。
            *   所有其他特定于行动的字段**必须**为空 (`""` 或 `[]`)。

        **Input:**
        *   **对话历史:**一个包含过去消息的字符串。
            {history}
        *   **情景快照：**一个描述当前情况的 JSON 对象。
            {snapshot_str}
        *   **可用工具：**一个列出可用工具及其定义的 JSON 对象。
            {tools_str}
        *   **可用策略：**一个包含可选策略的列表。
            {strategies}
    """)

    def create_rag_retrieval_prompt(self, knowledge_base_text: str, search_query: str) -> str:
        return textwrap.dedent(f"""
            You are a highly efficient information retrieval assistant. Your task is to extract the most relevant facts from the provided "Knowledge Base" that directly answer the "User's Search Query".

            - Respond ONLY with the extracted information.
            - If no relevant information is found, respond with the exact phrase "No relevant information found in the knowledge base."
            - Do not add any conversational text, greetings, or explanations.

            --- KNOWLEDGE BASE ---
            {knowledge_base_text}
            --- END KNOWLEDGE BASE ---

            User's Search Query: "{search_query}"
            """)

    def create_responder_prompt(self, user_input: str, snapshot: SituationalSnapshot, plan: ActionPlan, persona: Persona, rag_context: Optional[str] = None, tool_outputs: Optional[List[Dict]] = None) -> str:

        user_profile_str = snapshot.who_profile.model_dump_json(indent=2)
        action_plan_str = plan.model_dump_json(indent=2)

        reference_documents = ""
        if rag_context:
            reference_documents += f"**--- Retrieved Knowledge ---**\n{rag_context}\n\n"
        if tool_outputs:
            tool_outputs_str = json.dumps(
                tool_outputs, indent=2, ensure_ascii=False)
            reference_documents += f"**--- Tool Execution Results ---**\n{tool_outputs_str}\n\n"

        return textwrap.dedent(f"""
            ## Persona (Your Role) ##
            Adopt the following persona for your response. This is the most important instruction.
            **Name:** {persona.name}
            **Description:** {persona.description}
            Remember to maintain this persona throughout your entire answer.

            ## Role ##
            You are a top-tier, personalized AI assistant. Your task is to synthesize all available information to generate a final, helpful, and user-centric response while embodying your assigned persona.

            ## Context ##
            1.  **User's Latest Message:** `{user_input}`
            2.  **User Profile & Intent Analysis (for personalization and understanding unspoken needs):**
                ```json
                {user_profile_str}
                ```
                **Implicit Intents Guesses:** {snapshot.why_intent.model_dump_json(indent=2)}
            3.  **Action Plan (your guiding instruction):**
                ```json
                {action_plan_str}
                ```
            4.  **Reference Information (facts to use in your answer):**
                {reference_documents if reference_documents else "No external information was retrieved."}
            ## Your Task ##
            - **Embody Persona:** Strictly follow the persona description provided above.
            - **Synthesize:** Combine the information from the "Action Plan" and "Reference Information".
            - **Personalize:** Tailor your language based on the "User Profile". Your response should subtly address the user's likely implicit needs (see "Implicit Intents Guesses"), especially the high-confidence ones, making them feel truly understood.
            - **Respond:** Generate the final answer for the user.
            - **DO NOT** mention the user profile, action plan, persona, or any internal processes. Just provide the final, natural-sounding response.
            **Final Answer:**
            """)


class ToolManager:

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register_tool(self, tool_func: BaseTool):
        if not hasattr(tool_func, 'name'):
            raise ValueError("Tool function must have a 'name' attribute.")
        if tool_func.name in self._tools:
            logging.warning(
                f"Tool '{tool_func.name}' is already registered. Overwriting.")
        self._tools[tool_func.name] = tool_func
        logging.info(f"Tool '{tool_func.name}' registered.")

    def unregister_tool(self, tool_name: str):
        if tool_name in self._tools:
            del self._tools[tool_name]
            logging.info(f"Tool '{tool_name}' unregistered.")
        else:
            logging.warning(f"Tool '{tool_name}' not found for unregistering.")

    def get_all_tools(self) -> List[BaseTool]:
        return list(self._tools.values())

    def get_openai_tools(self) -> List[Dict]:
        """
        获取所有已注册工具并转换为OpenAI Function Calling格式。
        """
        return [
            {"type": "function", "function": convert_to_openai_function(t)}
            for t in self.get_all_tools()
        ]


MOCK_FINANCIAL_DATA = {
    "tencent": "609 billion Chinese yuan (Fiscal Year 2025)",
    "alibaba": "868.7 billion Chinese yuan (Fiscal Year 2023)",
    "bytedance": "120 billion US dollars (Fiscal Year 2023)"
}


def fetch_with_jina_reader(target_url: str, api_key: str) -> str | None:
    """
    使用 Jina Reader API 获取并返回给定URL的内容。

    这个函数模拟了以下 curl 命令：
    curl "https://r.jina.ai/{target_url}" -H "Authorization: Bearer {api_key}"

    参数:
    target_url (str): 你想要抓取内容的原始 URL。
    api_key (str): 你的 Jina API 密钥。

    返回:
    str: 成功时返回获取到的页面内容（通常是 Markdown 格式的文本）。
    None: 如果请求失败，则返回 None。
    """
    # 1. 拼接 Jina Reader API 的 URL
    base_url = "https://r.jina.ai/"
    full_url = f"{base_url}{target_url}"

    # 2. 准备请求头 (Headers)
    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    print(f"正在请求: {full_url}")

    try:
        # 3. 发送 GET 请求
        response = requests.get(
            full_url, headers=headers, timeout=60)  # 设置60秒超时

        # 4. 检查响应状态码，如果不是 200 (OK)，则抛出异常
        response.raise_for_status()

        # 5. 如果请求成功，返回响应的文本内容
        return response.text

    except requests.exceptions.HTTPError as e:
        # 处理 HTTP 错误 (例如 401 Unauthorized, 404 Not Found, 500 Server Error)
        print(f"发生 HTTP 错误: {e}")
        print(f"状态码: {e.response.status_code}")
        print(f"响应内容: {e.response.text}")
        return None
    except requests.exceptions.RequestException as e:
        # 处理其他网络相关的错误 (例如 DNS 查询失败，连接超时)
        print(f"发生请求错误: {e}")
        return None


@tool
def get_url_content(url: str):
    """
    获取并返回给定URL的主要文本内容。

    当用户提供一个链接并要求总结、提取信息或基于链接内容回答问题时，此工具非常有用。它使用Jina Reader API来提取网页的核心内容。

    Args:
        url: 需要读取的网页的完整URL (例如: 'https://www.example.com/article')。
    """
    key = os.getenv("JINA_API_KEY")
    if not key:
        return "Error: JINA_API_KEY environment variable is not set."
    content = fetch_with_jina_reader(url, key)
    if content is None:
        return f"Error: Failed to fetch content from {url}."
    return content


@tool
def get_company_revenue(company_name: str) -> str:
    """Gets the latest reported revenue for a given company."""
    name_lower = company_name.lower()
    return MOCK_FINANCIAL_DATA.get(name_lower, f"Sorry, I don't have financial data for {company_name}.")


@tool
def calculator(expression: str) -> str:
    """
    A calculator that evaluates a mathematical expression.
    Supports addition (+), subtraction (-), multiplication (*), and division (/).
    Example: '100 * (5 + 3)'
    """
    # 安全警告: 在生产环境中使用 eval() 是危险的，因为它可能执行任意代码。
    # 仅用于演示目的。在实际应用中，应使用更安全的解析器，如 numexpr 或 ast.literal_eval。
    try:
        allowed_chars = "0123456789+-*/(). "
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression."
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Error evaluating expression: {e}"


# ==============================================================================
# LangGraph 服务层
# ==============================================================================

class BaseAgent(ABC):
    @abstractmethod
    def invoke(self, query: str) -> Dict[str, Any]:
        """代理的统一入口点。"""
        pass


class SearchAgent(BaseAgent):
    """一个专门负责处理搜索任务的独立代理。"""

    class SearchAgentState(TypedDict):
        original_query: str
        refined_query: str
        search_results: List[BaseMessage]
        final_answer: str

    def __init__(self, client: OpenAIClient, search_tool: BaseTool):
        self.client = client
        self.tool_node = ToolNode([search_tool])
        workflow = StateGraph(self.SearchAgentState)
        workflow.add_node("refine_query", self._refine_query_node)
        workflow.add_node("execute_search", self._execute_search_node)
        workflow.add_node("synthesize_results", self._synthesize_results_node)

        workflow.set_entry_point("refine_query")
        workflow.add_edge("refine_query", "execute_search")
        workflow.add_edge("execute_search", "synthesize_results")
        workflow.add_edge("synthesize_results", END)

        self.graph = workflow.compile()

    def _execute_search_node(self, state: SearchAgentState) -> Dict[str, Any]:
        refined_query = state['refined_query']
        tool_call_message = AIMessage(
            content="",
            tool_calls=[{
                "name": "search_engine",
                "args": {"query": refined_query},
                "id": f"call_{uuid.uuid4()}"
            }]
        )
        search_results = state.get('search_results', [])
        results = self.tool_node.invoke([tool_call_message])

        search_results.extend(results)
        return {"search_results": search_results}

    def _refine_query_node(self, state: SearchAgentState) -> Dict[str, Any]:
        prompt = textwrap.dedent(f"""
            You are a search query refinement expert. Your task is to take a user's question and rephrase it into a concise, keyword-focused query suitable for a search engine.
            User Question: "{state['original_query']}"
            Refined Query:
        """)
        messages = [{"role": "system", "content": prompt}]
        response = self.client.get_chat_completion(
            messages, ANALYSIS_MODEL, 0.0)
        refined_query = response.get('content', state['original_query'])

        return {"refined_query": refined_query}

    def _synthesize_results_node(self, state: SearchAgentState) -> Dict[str, Any]:
        search_results_str = "\n".join(
            [res.content for res in state['search_results']])
        prompt = textwrap.dedent(f"""
            You are a synthesis expert. Based on the provided search results, answer the user's original question.
            Original Question: "{state['original_query']}"
            Search Results:
            ---
            {search_results_str}
            ---
            Synthesized Answer:
        """)
        messages = [{"role": "system", "content": prompt}]
        response = self.client.get_chat_completion(
            messages, RESPONSE_MODEL, 0.2)
        final_answer = response.get(
            'content', "Could not synthesize an answer.")
        return {"final_answer": final_answer}

    def invoke(self, query: str) -> Dict[str, Any]:
        """执行搜索代理工作流。"""
        return self.graph.invoke({"original_query": query})


MOCK_SEARCH_RESULTS = {
    "langgraph": "LangGraph is a library for building stateful, multi-actor applications with LLMs. It extends the LangChain expression language with the ability to coordinate multiple chains (or actors) across multiple steps of computation in a cyclical manner.",
    "rag": "Retrieval-Augmented Generation (RAG) is an AI framework for improving the quality of LLM-generated responses by grounding the model on external sources of knowledge to supplement the LLM’s internal representation of information.",
    "agent memory": "Agent memory is the capability of an AI agent to retain and recall information from past interactions. This allows the agent to maintain context, learn from experience, and provide more personalized and coherent responses over time."
}


@tool
def search_engine(query: str) -> str:
    """
    Simulates a search engine to find information on the web about a given topic.
    Useful for finding definitions or explanations of concepts like 'LangGraph', 'RAG', or 'Agent Memory'.
    """
    query_lower = query.lower()
    for keyword, result in MOCK_SEARCH_RESULTS.items():
        if keyword in query_lower:
            return f"Search results for '{query}':\n{result}"
    return f"No specific information found for your query '{query}'. Try searching for 'LangGraph', 'RAG', or 'Agent Memory'."


class KnowledgeBase:
    """
    一个可扩展的知识库类，用于管理和检索信息。
    这个基础实现使用一个硬编码的文本块作为知识源。
    可以被继承以实现更复杂的检索逻辑，例如从数据库或向量存储中检索。
    """

    def __init__(self):
        # 将mock数据移到这里
        self._knowledge_text = "\n".join([
            "Fact: Tencent is a leading internet and technology company based in China.",
            "Fact: Alibaba Group is a Chinese multinational technology company specializing in e-commerce, retail, Internet, and technology.",
            "Fact: Alibaba Cloud is a subsidiary of Alibaba Group and is the largest cloud computing company in China.",
            "Fact: You should replace the knowledge base with your own relevant documents for best results.",
        ])
        logging.info("知识库已初始化（使用内存中的mock数据）。")

    def get_content(self) -> str:
        """
        返回知识库的全部内容。
        在更高级的实现中，这里可以是基于查询的检索。
        """
        return self._knowledge_text


class LangGraphService:

    def __init__(self, client: OpenAIClient, prompter: PromptFactory, history_manager: HistoryManager, tool_manager: ToolManager, knowledge_base: KnowledgeBase, search_agent: BaseAgent):
        self.client = client
        self.prompter = prompter
        self.history_manager = history_manager
        self.tool_manager = tool_manager
        self.knowledge_base = knowledge_base
        self.openai_tools = self.tool_manager.get_openai_tools()
        self.tool_node = ToolNode(self.tool_manager.get_all_tools())

        workflow = StateGraph(GraphState)
        workflow.add_node("manage_history", self.manage_history_node)
        workflow.add_node("analyze_situation", self.analyze_situation_node)

        workflow.add_node("update_profile", self.update_profile_node)
        workflow.add_node("determine_strategy", self.determine_strategy_node)
        workflow.add_node("retrieve_knowledge", self.retrieve_knowledge_node)
        workflow.add_node("generate_response", self.generate_response_node)
        workflow.add_node("execute_tools", self.tool_executor)
        workflow.add_node("delegate_to_search_agent",
                          self.delegate_to_search_agent_node)
        workflow.add_node("describe_image", self.describe_image_node)

        workflow.set_entry_point("manage_history")

        workflow.add_conditional_edges(
            "manage_history",
            lambda state: "describe_image" if state.get(
                "image_b64_urls") else "analyze_situation",
            {
                "describe_image": "describe_image",
                "analyze_situation": "analyze_situation"
            }
        )

        workflow.add_edge("describe_image", "analyze_situation")

        workflow.add_conditional_edges(
            "analyze_situation",
            self.route_after_analysis,
            {
                "update_profile": "update_profile",
                "continue": "determine_strategy"
            }
        )

        workflow.add_edge("update_profile",
                          "determine_strategy")

        workflow.add_conditional_edges(
            "determine_strategy",
            self.route_after_strategy,
            {
                "rag": "retrieve_knowledge",
                "tool": "execute_tools",
                "direct": "generate_response",
                "search_agent": "delegate_to_search_agent",
            }
        )

        workflow.add_edge("retrieve_knowledge", "generate_response")
        workflow.add_edge("execute_tools", "generate_response")
        workflow.add_edge("delegate_to_search_agent", "generate_response")
        workflow.add_edge("generate_response", END)

        self.graph = workflow.compile()

        self.search_agent = search_agent

    def describe_image_node(self, state: GraphState) -> Dict[str, Any]:
        """如果提供了图片，则调用多模态模型生成统一描述并添加到历史记录中。"""
        image_b64_list = state.get("image_b64_urls")
        if not image_b64_list:
            logging.info("未提供图片，跳过图片描述节点。")
            return {}

        logging.info(f"检测到 {len(image_b64_list)} 张图片，开始生成统一描述...")
        prompt = "请详细描述以下所有图片的内容。如果图片之间有关联，请说明它们的关系。请综合所有信息，给出一个连贯的描述。"
        description = self.client.describe_image(
            image_b64_list, prompt, VISION_MODEL)

        logging.info(f"图片描述生成: {description}")

        # 将图片描述作为一条工具消息注入到对话历史中
        image_context_message = ToolMessage(
            name="上传的图片描述",
            tool_call_id=str(uuid.uuid4()),
            content=f"用户提供了 {len(image_b64_list)} 张图片，其综合内容描述如下：\n---\n{description}\n---"
        )
        return {"conversation_history": [image_context_message], "image_b64_urls": []}

    def delegate_to_search_agent_node(self, state: GraphState) -> Dict[str, Any]:
        logging.info("委托任务给搜索代理...")
        original_query = state['plan'].search_query

        agent_result = self.search_agent.invoke(original_query)

        synthesized_answer = agent_result.get(
            'final_answer', "搜索代理未能返回结果。")
        logging.info(f"搜索代理返回结果: {synthesized_answer}")

        return {"rag_context": synthesized_answer}

    def tool_executor(self, state: GraphState) -> GraphState:
        tool_calls_message = state['conversation_history'][-1]
        result = self.tool_node.invoke([tool_calls_message])
        state['conversation_history'].extend(result)
        return state

    def _clean_and_parse_json(self, raw_response: str) -> Optional[Dict[str, Any]]:
        match = re.search(r"```json\s*([\s\S]*?)\s*```", raw_response)
        if match:
            json_str = match.group(1)
        else:
            start = raw_response.find('{')
            end = raw_response.rfind('}')
            if start != -1 and end != -1 and end > start:
                json_str = raw_response[start:end+1]
            else:
                json_str = raw_response
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logging.error(f"JSON解析失败: {e}\n原始字符串: '{json_str}'")
            return None

    def _call_llm_for_json_content(self, prompt: str, model: str, temperature: float) -> Optional[Dict[str, Any]]:
        messages = [{"role": "system", "content": prompt}]
        response_message = self.client.get_chat_completion(
            messages, model, temperature, is_json=True)
        content = response_message.get("content", "")
        if not content:
            return {"error": "LLM returned an empty response."}
        return self._clean_and_parse_json(content)

    def manage_history_node(self, state: GraphState) -> Dict[str, Any]:
        logging.info("管理对话历史节点被触发。")
        history = state['conversation_history']
        compressed_history = self.history_manager.compress_history_if_needed(
            history)
        return {"conversation_history": compressed_history}

    def _deep_merge_dicts(self, base: dict, update: dict) -> dict:
        merged = deepcopy(base)
        for key, value in update.items():
            if key in merged and isinstance(merged.get(key), dict) and isinstance(value, dict):
                # 递归合并嵌套字典
                merged[key] = self._deep_merge_dicts(merged[key], value)

            elif key in merged and isinstance(merged.get(key), list) and isinstance(value, list):
                # --- 开始：列表合并的智能逻辑 ---
                base_list = merged[key]
                update_list = value

                # 检查列表是否包含字典，以决定合并策略
                is_list_of_dicts = (base_list and isinstance(base_list[0], dict)) or \
                                   (update_list and isinstance(
                                       update_list[0], dict))

                if is_list_of_dicts:
                    # 策略1: 合并字典列表 (Upsert logic)
                    # 尝试为对象找到一个唯一键（如ID或名称）
                    potential_unique_keys = [
                        'skillName', 'item', 'title', 'name', 'id']
                    unique_key = None

                    sample_item = next(
                        (item for item in base_list + update_list if item), None)
                    if sample_item:
                        for pk in potential_unique_keys:
                            if pk in sample_item:
                                unique_key = pk
                                break

                    if unique_key:
                        # 使用字典进行高效的更新或插入操作
                        merged_map = {
                            item[unique_key]: item for item in base_list if unique_key in item}
                        for item in update_list:
                            if unique_key in item:
                                merged_map[item[unique_key]] = item  # 覆盖或添加
                        merged[key] = list(merged_map.values())
                    else:
                        # 如果找不到唯一键，则直接追加新项目（避免去重）
                        merged[key].extend(update_list)
                else:
                    # 策略2: 合并简单类型（如字符串）的列表并去重
                    try:
                        combined_list = base_list + update_list
                        # 使用 dict.fromkeys 安全去重
                        merged[key] = list(dict.fromkeys(combined_list))
                    except TypeError:
                        # 如果遇到其他不可哈希类型，则直接追加
                        merged[key].extend(update_list)
                # --- 结束：列表合并的智能逻辑 ---
            else:
                # 对于其他所有情况（新键或类型不匹配），直接覆盖
                merged[key] = value
        return merged

    def analyze_situation_node(self, state: GraphState) -> Dict[str, Any]:
        logging.info("情景分析节点被触发。")
        prompt = self.prompter.create_situational_analysis_prompt(
            state['user_input'], state['conversation_history'], state['user_profile'])
        data = self._call_llm_for_json_content(prompt, ANALYSIS_MODEL, 0.1)

        if not data:
            data = {}

        try:
            analysis_output = SituationalAnalysisOutput.model_validate(data)
            snapshot = SituationalSnapshot(
                who_profile=state['user_profile'],
                what_topic=analysis_output.what_topic_analysis,
                why_intent=analysis_output.why_intent_analysis,
                how_demand=analysis_output.how_demand
            )
            return {"snapshot": snapshot}
        except ValidationError as e:
            logging.error(f"情景分析节点 Pydantic 校验失败: {e}\n接收到的数据: {data}")
            snapshot = SituationalSnapshot(who_profile=state['user_profile'], what_topic=TopicAnalysis(
            ), why_intent=IntentAnalysis(), how_demand=CoreDemand())
            return {"snapshot": snapshot}

    def update_profile_node(self, state: GraphState) -> Dict[str, Any]:
        logging.info("周期性用户画像更新节点被触发。")
        prompt = self.prompter.create_profile_update_prompt(
            state['conversation_history'], state['user_profile'])

        profile_update_data = self._call_llm_for_json_content(
            prompt, PROFILE_MODEL, 0.2)

        if not profile_update_data or "error" in profile_update_data:
            logging.warning("画像更新 LLM 调用失败或返回空，跳过更新。")
            return {}  # 不做任何修改

        try:
            # 不再需要 sanitize，直接合并
            current_profile_dict = state['user_profile'].model_dump()
            updated_profile_dict = self._deep_merge_dicts(
                current_profile_dict, profile_update_data)  # 直接使用LLM的输出
            updated_profile = UserProfile.model_validate(updated_profile_dict)

            logging.info("用户画像更新成功。")
            # 更新快照中的画像部分，以便后续节点能用到最新的画像
            updated_snapshot = state['snapshot'].model_copy(
                update={'who_profile': updated_profile})

            return {"user_profile": updated_profile, "snapshot": updated_snapshot}
        except ValidationError as e:
            logging.error(
                f"画像更新 Pydantic 校验失败: {e}\n接收到的数据: {profile_update_data}")
            return {}  # 校验失败，保持原画像

    def format_history_for_prompt(self, history: List[BaseMessage]) -> str:
        def get_text_from_content(content):
            if isinstance(content, str):
                return content
            elif isinstance(content, list):
                return " ".join(
                    part["content"] for part in content if part["type"] == "text"
                )
            return ""
        history_str = "\n".join(
            f"{MESSAGE_TYPE[msg.type]}: {get_text_from_content(msg.content)}" for msg in history)
        return history_str

    def determine_strategy_node(self, state: GraphState) -> Dict[str, Any]:

        logging.info("进入 determine_strategy_node，开始制定行动策略。")

        history = state['conversation_history'][-6:]  # 仅取最近6条消息
        history_str = self.history_manager.format_history_for_model(history)

        prompt = self.prompter.create_strategy_prompt(
            state['snapshot'], history_str, self.tool_manager.get_openai_tools())
        data = self._call_llm_for_json_content(prompt, STRATEGY_MODEL, 0.3)

        if not data:
            plan = ActionPlan(strategy=ActionStrategy.DIRECT_ANSWER, reasoning="Fallback strategy due to error.",
                              proposed_response_outline="Address user directly.")
            return {"plan": plan}

        try:
            plan = ActionPlan.model_validate(data)

            if plan.strategy == ActionStrategy.TOOL_USE and plan.tool_calls:

                tool_calls_for_message = [
                    {
                        "name": tc.name,
                        "args": tc.parameters,
                        "id": f"call_{uuid.uuid4()}"
                    } for tc in plan.tool_calls
                ]

                ai_message_with_tool_calls = AIMessage(
                    content="",
                    tool_calls=tool_calls_for_message
                )
                return {"plan": plan, "conversation_history": [ai_message_with_tool_calls]}

            return {"plan": plan}
        except ValidationError as e:
            logging.error(f"策略节点 Pydantic 校验失败: {e}")
            plan = ActionPlan(strategy=ActionStrategy.DIRECT_ANSWER, reasoning="Fallback strategy due to validation error.",
                              proposed_response_outline="Address user directly.")
            return {"plan": plan}

    def retrieve_knowledge_node(self, state: GraphState) -> Dict[str, Any]:
        search_query = state['plan'].search_query
        knowledge_content = self.knowledge_base.get_content()
        prompt = self.prompter.create_rag_retrieval_prompt(
            knowledge_content, search_query)
        messages = [{"role": "system", "content": prompt}]
        response_message = self.client.get_chat_completion(
            messages, RETRIEVAL_MODEL, 0.0)
        retrieved_text = response_message.get(
            "content", "Failed to retrieve from knowledge base.")
        return {"rag_context": retrieved_text}

    def generate_response_node(self, state: GraphState) -> Dict[str, Any]:
        logging.info("进入 generate_response_node，开始合成最终回复。")
        tool_outputs = []
        for msg in reversed(state['conversation_history']):
            if isinstance(msg, ToolMessage) or isinstance(msg, FunctionMessage):
                tool_outputs.append(
                    {"tool_name": msg.name, "output": msg.content})
            elif isinstance(msg, ChatMessage) and msg.tool_calls:
                break

        prompt = self.prompter.create_responder_prompt(
            state['user_input'],
            state['snapshot'],
            state['plan'],
            state['active_persona'],
            state.get('rag_context'),
            tool_outputs if tool_outputs else None
        )

        messages = [{"role": "system", "content": prompt}]

        # 使用非JSON模式调用，因为我们期望的是自然语言文本
        response_message = self.client.get_chat_completion(
            messages, RESPONSE_MODEL, 0.7, is_json=False)

        final_response = response_message.get("content", "抱歉，我无法生成回复。")

        # 将最终回复包装成 AIMessage 添加到历史记录
        final_ai_message = AIMessage(content=final_response)

        return {
            "final_response": final_response,
            "conversation_history": [final_ai_message]
        }

    def route_after_analysis(self, state: GraphState) -> str:
        turn_count = state.get('turn_count', 0)
        if turn_count > 0 and turn_count % PROFILE_UPDATE_INTERVAL == 0:
            logging.info(f"第 {turn_count} 轮对话：路由到 'update_profile'。")
            return "update_profile"
        else:
            logging.info(f"第 {turn_count} 轮对话：路由到 'determine_strategy'。")
            return "continue"

    def route_after_strategy(self, state: GraphState) -> str:

        strategy = state['plan'].strategy
        logging.info(f"路由决策：检测到策略为 '{strategy.value}'。")

        if strategy == ActionStrategy.RAG_QUERY:
            logging.info("路由到 'retrieve_knowledge' (rag)。")
            return "rag"
        elif strategy == ActionStrategy.TOOL_USE and state['plan'].tool_calls:
            logging.info("路由到 'execute_tools' (tool)。")
            return "tool"
        elif strategy == ActionStrategy.SEARCH_AGENT_DELEGATION:
            return "search_agent"
        else:  # DIRECT_ANSWER, CLARIFY_ASK, PROACTIVE_GUIDE
            logging.info("路由到 'generate_response' (direct)。")
            return "direct"

    def stream(self, **kwargs):
        """Wrapper for the graph stream method."""
        return self.graph.stream(kwargs)
# ==============================================================================
# UI 层 (Gradio Application)
# ==============================================================================


class ChatApplication:
    def __init__(self, service: LangGraphService, session_manager: SessionManager, persona_manager: PersonaManager):
        self.service = service
        self.session_manager = session_manager
        self.persona_manager = persona_manager

    def _handle_file_upload(self, file):
        """当文件上传时，更新按钮状态以提供反馈。"""
        if file:
            return gr.update(interactive=False)
        return gr.update()

    def _convert_langchain_to_gradio(self, messages: List[BaseMessage]) -> List[Dict[str, any]]:
        history = []

        def concat_human_message(msg: HumanMessage):
            content = msg.content
            if isinstance(msg.content, str):
                return msg.content
            elif isinstance(msg.content, list):
                text_for_display = ""
                for part in content:
                    if part["type"] == "text":
                        text_for_display = part["content"]
                    elif part["type"] == "image":
                        text_for_display += f'<img src="{part["content"]}" style="max-width:300px;"/>'
                return text_for_display
            return ""

        for msg in messages:
            if not (isinstance(msg, (HumanMessage, AIMessage)) and msg.content):
                continue

            if isinstance(msg, HumanMessage):
                history.append(
                    {"role": "user", "content": concat_human_message(msg)})

            elif isinstance(msg, AIMessage):
                history.append({"role": "assistant", "content": msg.content})

        return history

    def add_new_session(self):
        session_id, _ = self.session_manager.create_session()
        session_list = self.session_manager.get_session_list()
        return (
            gr.Radio(choices=session_list, value=session_id),
            session_id,
            [],
            gr.update(value={}),
            gr.update(value="...")
        )

    def update_session_persona(self, session_id: str, persona_name: str):
        if not session_id:
            return
        self.session_manager.update_persona_for_session(
            session_id, persona_name)

    def switch_session(self, selected_session_id: str):
        if not selected_session_id:
            default_persona = self.persona_manager.get_default_persona().name
            return None, [], gr.update(value={}), gr.update(value="..."), gr.update(value=default_persona)

        session_data = self.session_manager.get_session_data(
            selected_session_id)

        history = []
        persona_name = self.persona_manager.get_default_persona().name

        if session_data:
            history = self._convert_langchain_to_gradio(
                session_data.get("conversation_history", []))
            persona_name = session_data.get(
                "active_persona_name", persona_name)

        return (
            selected_session_id,
            history,
            gr.update(value={}),
            gr.update(value="..."),
            gr.update(value=persona_name)
        )

    def delete_selected_session(self, session_to_delete_id: str):
        if not session_to_delete_id:
            return gr.update(), None, [], gr.update(value={}), gr.update(value="..."), gr.update()

        self.session_manager.delete_session(session_to_delete_id)
        session_list = self.session_manager.get_session_list()
        new_selected_id = session_list[0][1] if session_list else None

        history = []
        persona_name = self.persona_manager.get_default_persona().name

        if new_selected_id:
            session_data = self.session_manager.get_session_data(
                new_selected_id)
            if session_data:
                history = self._convert_langchain_to_gradio(
                    session_data.get("conversation_history", []))
                persona_name = session_data.get(
                    "active_persona_name", persona_name)

        return (
            gr.Radio(choices=session_list, value=new_selected_id),
            new_selected_id,
            history,
            gr.update(value={}),
            gr.update(value="..."),
            gr.update(value=persona_name)
        )

    def chat_flow(self, user_input: str, uploaded_file: Optional[str], current_session_id: str) -> Iterator:
        if (not user_input or not user_input.strip()) and not uploaded_file:
            yield [], gr.update(), gr.update()
            return

        if not current_session_id:
            raise gr.Error(
                "No active session. Please create one using the '+ New Session' button.")

        session_data = self.session_manager.get_session_data(
            current_session_id)
        if not session_data:
            raise gr.Error(
                f"Critical error: Could not find data for session ID {current_session_id}")

        llm_history = session_data.get("conversation_history", [])
        display_history = self._convert_langchain_to_gradio(llm_history)

        image_b64_urls = []
        ui_image_parts = []

        url_matches = re.findall(
            r'https?://\S+\.(?:jpg|jpeg|png|gif|webp|bmp)', user_input, re.IGNORECASE)
        if url_matches:
            for url in url_matches:
                image_data = self._get_image_from_url(url)
                if image_data:
                    image_bytes, mime_type = image_data
                    b64_string = base64.b64encode(image_bytes).decode('utf-8')
                    data_url = f"data:{mime_type};base64,{b64_string}"
                    image_b64_urls.append(data_url)
                    ui_image_parts.append({"type": "image", "image": data_url})

        if uploaded_file:
            try:
                with open(uploaded_file, "rb") as f:
                    image_bytes = f.read()
                mime_type, _ = mimetypes.guess_type(uploaded_file) or (
                    "application/octet-stream", None)
                b64_string = base64.b64encode(image_bytes).decode('utf-8')
                data_url = f"data:{mime_type};base64,{b64_string}"
                image_b64_urls.append(data_url)
                ui_image_parts.append({"type": "image", "image": data_url})
            except Exception as e:
                logging.error(f"Error processing uploaded file: {e}")

        text_for_display = ""
        content_for_msg = []
        for img in image_b64_urls:
            text_for_display += f'<img src="{img}" style="max-width:300px;"/>'
            content_for_msg.append({"type": "image", "content": img})
        content_for_llm = user_input.strip()
        if not content_for_llm and image_b64_urls:
            content_for_llm = "请详细描述图片内容。"
            user_input = content_for_llm
        else:
            text_for_display += f"\n{user_input.strip()}"
            content_for_msg.append(
                {"type": "text", "content": user_input.strip()})

        if len(content_for_msg) == 1 and content_for_msg[0]["type"] == "text":
            content_for_msg = content_for_msg[0]["content"]



        if text_for_display:
            display_history.append(
                {"role": "user", "content": text_for_display})
            llm_history.append(HumanMessage(content=content_for_msg))

        display_history.append(
            {"role": "assistant", "content": "🧠 Thinking..."})
        yield display_history, gr.update(value={}), gr.update(value="Thinking...")

        active_persona_name = session_data.get(
            "active_persona_name", self.persona_manager.get_default_persona().name)
        active_persona = self.persona_manager.get_persona(active_persona_name)

        graph_input = {
            "conversation_history": llm_history,
            "user_profile": session_data["user_profile"],
            "user_input": user_input,
            "turn_count": session_data["turn_count"],
            "active_persona": active_persona,
        }
        
        
        
        if image_b64_urls:
            graph_input["image_b64_urls"] = image_b64_urls

        final_state, snapshot_json, plan_md = {}, {}, "..."
        for event in self.service.stream(**graph_input):
            node_name = list(event.keys())[0]
            node_output = event[node_name]
            if node_output:
                final_state.update(node_output)
            else:
                continue
            assistant_status_message = display_history[-1]["content"]

            if node_name == 'describe_image' and 'conversation_history' in node_output:
                assistant_status_message = "🖼️ Analyzing image(s)..."

            if 'snapshot' in node_output:
                assistant_status_message = "🤔 Analyzing context..."
                snapshot_json = final_state['snapshot'].model_dump(
                    exclude_unset=True)

            if node_name == 'update_profile' and 'user_profile' in node_output:
                assistant_status_message = "🔄 Consolidating user profile..."
                if 'snapshot' in node_output:
                    snapshot_json = node_output['snapshot'].model_dump_json(
                        indent=2, exclude_unset=True)

            if 'plan' in node_output:
                assistant_status_message = "🎯 Deciding strategy..."
                plan = final_state['plan']
                plan_md = f"**Strategy:** `{plan.strategy.value}`\n\n**Reasoning:**\n{plan.reasoning}"

            if 'rag_context' in node_output:
                plan_md += f"\n\n**Retrieved Context:**\n```\n{node_output['rag_context']}\n```"
                assistant_status_message = "📚 Retrieving knowledge..."

            if node_name == 'execute_tools':
                tool_output_messages = node_output.get(
                    'conversation_history', [])
                if tool_output_messages:
                    messages_to_display = tool_output_messages[-2:]
                    display_count = len(messages_to_display)
                    plan_md += f"\n\n**Tool Output (Last {display_count}):**\n```json\n{json.dumps([m.model_dump() for m in messages_to_display], indent=2, ensure_ascii=False)}\n```"
                assistant_status_message = "⚙️ Executing tools..."

            display_history[-1]["content"] = assistant_status_message
            yield display_history, gr.update(value=snapshot_json), gr.update(value=plan_md)

        final_response = final_state.get(
            "final_response", "Sorry, an error occurred.")
        display_history[-1]["content"] = final_response

        final_history = llm_history + \
            final_state.get('conversation_history', [])[-1:]
        final_state['conversation_history'] = final_history
        self.session_manager.update_session_data(
            current_session_id, final_state)

        yield display_history, gr.update(value=snapshot_json), gr.update(value=plan_md)

    def _get_image_from_url(self, url: str) -> Optional[Tuple[bytes, str]]:
        """从URL下载图片并返回其字节和MIME类型。"""
        try:
            # 添加常见的User-Agent，防止一些网站阻止请求
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
            response = requests.get(
                url, stream=True, timeout=10, headers=headers)
            response.raise_for_status()
            
            content_type = response.headers.get('Content-Type')
            if content_type and content_type.startswith('image'):
                return response.content, content_type
            logging.warning(f"URL {url} 的 Content-Type ({content_type}) 不是图片。")
            return None
        except requests.exceptions.RequestException as e:
            logging.error(f"从URL下载图片失败: {e}")
            return None

    def launch(self):
        css = """
        #chatbot { min-height: 75vh; }
        """
        sessions = self.session_manager.get_session_list()
        initial_session_id = sessions[0][1] if sessions else None
        with gr.Blocks(theme=gr.themes.Soft(), css=css) as demo:
            current_session_id_state = gr.State(initial_session_id)

            gr.Markdown("# 🤖 更懂你的AI")

            with gr.Row():
                with gr.Column(scale=1, min_width=80) as sidebar_col:

                    with gr.Group(visible=True) as expanded_sidebar:
                        with gr.Row(equal_height=False):
                            gr.Markdown("### 会话管理")
                            collapse_btn = gr.Button(
                                "ᐊ 折叠", size="sm", variant="ghost", min_width=60)

                        initial_session_data = self.session_manager.get_session_data(
                            initial_session_id)
                        initial_persona = initial_session_data.get(
                            "active_persona_name") if initial_session_data else self.persona_manager.get_default_persona().name
                        persona_dd = gr.Dropdown(
                            label="AI人格",
                            choices=self.persona_manager.list_persona_names(),
                            value=initial_persona,
                            interactive=True
                        )
                        gr.Markdown("---")
                        add_session_btn = gr.Button(
                            "+ 新建会话", variant="primary", size="sm")
                        initial_session_list = self.session_manager.get_session_list()
                        session_list_radio = gr.Radio(
                            label="当前会话列表",
                            choices=initial_session_list,
                            value=initial_session_id,
                            interactive=True
                        )
                        delete_session_btn = gr.Button("删除选中会话", size="sm")

                    with gr.Group(visible=False) as collapsed_sidebar:
                        expand_btn = gr.Button("ᐅ", size="sm", variant="ghost")

                with gr.Column(scale=8) as main_col:
                    with gr.Row():
                        with gr.Column(scale=2):
                            chatbot = gr.Chatbot(
                                label="对话窗口", elem_id="chatbot", show_copy_button=True, type='messages'
                            )
                            with gr.Row():

                                upload_btn = gr.UploadButton(
                                    "🖼️ 上传图片", file_types=["image"], elem_id="upload_button"
                                )
                                user_input_textbox = gr.Textbox(
                                    placeholder="在这里输入你的消息...", label="你的消息", container=False, scale=4
                                )
                                send_btn = gr.Button(
                                    "发送", variant="primary", scale=1)
                        with gr.Column(scale=1, min_width=300):
                            gr.Markdown("### 🧠 AI 在想什么")
                            with gr.Accordion("情景快照 & 用户画像", open=True):
                                snapshot_view = gr.JSON(label="分析结果", value={})
                            with gr.Accordion("行动规划 & 知识/工具/搜索", open=True):
                                plan_view = gr.Markdown(value="等待输入...")

            def collapse_sidebar():
                return (
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(scale=0, min_width=60)
                )

            def expand_sidebar():
                return (
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(scale=1, min_width=250)
                )

            collapse_btn.click(
                fn=collapse_sidebar,
                inputs=None,
                outputs=[expanded_sidebar, collapsed_sidebar, sidebar_col]
            )
            expand_btn.click(
                fn=expand_sidebar,
                inputs=None,
                outputs=[expanded_sidebar, collapsed_sidebar, sidebar_col]
            )

            persona_dd.change(
                fn=self.update_session_persona,
                inputs=[current_session_id_state, persona_dd],
                outputs=None
            )

            def add_new_session_ui_update():
                session_id, _ = self.session_manager.create_session()
                session_list = self.session_manager.get_session_list()
                default_persona = self.persona_manager.get_default_persona().name
                return (
                    gr.Radio(choices=session_list, value=session_id),
                    session_id,
                    [], gr.update(value={}), gr.update(value="..."),
                    gr.Dropdown(value=default_persona)  # Reset dropdown
                )

            ui_outputs = [session_list_radio, current_session_id_state,
                          chatbot, snapshot_view, plan_view, persona_dd]
            add_session_btn.click(
                add_new_session_ui_update, outputs=ui_outputs)
            session_list_radio.change(inputs=[session_list_radio], outputs=[
                                      current_session_id_state, chatbot, snapshot_view, plan_view, persona_dd])
            delete_session_btn.click(self.delete_selected_session, inputs=[
                                     current_session_id_state], outputs=ui_outputs)

            chat_inputs = [user_input_textbox,
                           upload_btn, current_session_id_state]
            chat_outputs = [chatbot, snapshot_view, plan_view]

            upload_btn.upload(
                self._handle_file_upload,
                inputs=[upload_btn],
                outputs=[upload_btn]
            )

            def clear_and_reenable_inputs():
                return "", gr.update(value=None, interactive=True)

            send_btn.click(
                self.chat_flow,
                inputs=chat_inputs,
                outputs=chat_outputs
            ).then(
                clear_and_reenable_inputs,
                outputs=[user_input_textbox, upload_btn]
            )

            user_input_textbox.submit(
                self.chat_flow,
                inputs=chat_inputs,
                outputs=chat_outputs
            ).then(
                clear_and_reenable_inputs,
                outputs=[user_input_textbox, upload_btn]
            )
        demo.launch()


# ==============================================================================
# 主程序入口
# ==============================================================================
if __name__ == "__main__":
    try:

        openai_client = OpenAIClient(api_key=API_KEY, base_url=BASE_URL)
        prompt_factory = PromptFactory()
        history_manager = HistoryManager(client=openai_client)
        persona_manager = PersonaManager()
        tool_manager = ToolManager()
        tool_manager.register_tool(get_company_revenue)
        tool_manager.register_tool(calculator)
        tool_manager.register_tool(get_url_content)
        knowledge_base = KnowledgeBase()
        search_agent = SearchAgent(
            client=openai_client, search_tool=search_engine)

        langgraph_service = LangGraphService(
            client=openai_client,
            prompter=prompt_factory,
            history_manager=history_manager,
            tool_manager=tool_manager,
            knowledge_base=knowledge_base,
            search_agent=search_agent,
        )

        session_manager = SessionManager()

        session_list = session_manager.get_session_list()
        if not session_list:
            logging.info("自动创建一个默认的会话。")
            initial_session_id, _ = session_manager.create_session()

        app = ChatApplication(service=langgraph_service,
                              session_manager=session_manager,
                              persona_manager=persona_manager,)
        app.launch()

    except ValueError as e:
        print(f"配置错误: {e}")
    except Exception as e:
        logging.critical(f"应用程序启动失败: {e}", exc_info=True)
        print(
            f"FATAL: Application failed to start. Check logs for details. Error: {e}")
