import os
# 파일 감시 관련 에러 방지를 위한 환경 변수 설정
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import datetime
import json
import traceback
from typing import List, Dict, Any, Tuple, Optional
import io
import gc

# --- 모델 응답 시뮬레이션 효과를 위한 함수 ---
def simulate_typing(response, placeholder):
    """모델 응답이 타이핑되는 효과를 구현합니다."""
    full_response = ""
    
    # 응답을 한 글자씩 표시
    for i in range(min(len(response), 20)):  # 처음 20자는 개별 표시
        full_response += response[i]
        placeholder.markdown(full_response + "▌")
        time.sleep(0.01)  # 짧은 지연
    
    # 나머지 내용은 청크 단위로 표시
    if len(response) > 20:
        chunk_size = max(5, len(response) // 20)  # 남은 텍스트 길이에 따라 청크 크기 조정
        for i in range(20, len(response), chunk_size):
            chunk = response[i:i+chunk_size]
            full_response += chunk
            placeholder.markdown(full_response + "▌")
            time.sleep(0.01)
    
    # 최종 응답 표시
    placeholder.markdown(full_response)
    return full_response

# --- 메모리 사용량 관리 함수 ---
def clear_cuda_cache():
    """CUDA 캐시를 비우고 가비지 컬렉션을 실행하여 메모리를 확보합니다."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

# --- Streamlit 페이지 설정 (가장 먼저 호출되어야 함) ---
st.set_page_config(
    page_title="청소년 마음 상담 챗봇", 
    page_icon="🧑‍⚕️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 안정적인 기본값 설정 ---
# 모델 경로 설정
model_path = "./final_model"
device = "cuda" if torch.cuda.is_available() else "cpu"
# 기본 토큰 길이 제한 (안전한 값으로 설정)
DEFAULT_MAX_NEW_TOKENS = 512
MAX_CONTEXT_LENGTH = 8192  # 컨텍스트 창 크기를 안전한 값으로 제한

# 시스템 프롬프트
SYSTEM_PROMPT = """
당신은 공감 능력이 뛰어난 전문 청소년 상담사입니다.
다음과 같은 지침을 따르십시오:

📌 역할 및 태도
당신은 심리학 및 청소년 발달에 정통한 전문가입니다.
청소년 내담자가 심리적 안전감을 느낄 수 있도록, 따뜻하고 부드러운 언어로 대화합니다.
내담자의 말에 비판하거나 판단하지 않으며, 공감하고 경청합니다.
복잡한 조언보다는 공감적 반응과 열린 질문을 통해 내담자가 스스로 이야기할 수 있도록 유도합니다.

💬 언어 및 문체
말투는 친근하고 편안한 어조로 유지하되, 너무 캐주얼하거나 유행어는 피합니다.
"그랬구나", "그럴 수 있어", "네가 그런 기분을 느낀 건 충분히 이해돼" 같은 공감 표현을 자주 사용하세요.
질문은 "~한 적 있니?", "어떤 생각이 들었을까?"처럼 자기 탐색을 유도하는 방식으로 구성하세요.

⛔ 금지사항
진단, 약물 추천, 명확한 조언, 법적 판단은 절대 제공하지 않습니다.
자해, 자살, 폭력 등 위기 상황은 **"신뢰할 수 있는 어른이나 전문가에게 꼭 도움을 요청하길 바란다"**고 안내하세요.
내담자를 통제하거나 설득하려 하지 않습니다.

🧭 대화 방식 예시
공감 → 탐색 → 확장 or 요약의 흐름으로 진행합니다.
내담자의 감정에 이름을 붙여주는 감정 라벨링도 적절히 사용하세요.
내담자가 말이 없거나 모호하게 표현할 경우, 부드러운 재질문이나 구체적인 예시로 도와줍니다.

🎯 대화 목적
내담자가 스스로 자신의 감정과 상황을 정리하고 이해하도록 돕는 것이 가장 큰 목적입니다.
문제 해결이 아닌, 정서적 지지와 자기 이해가 중심입니다.

초기 대화시 내담자의 이름을 먼저 물어보고, 이름을 불러주면서 공감하고 경청합니다.
"""

# 상수 정의
MAX_HISTORY_LENGTH = 20  # 최대 대화 기록 길이 (메모리 사용량을 고려한 안전한 값)

# --- 로깅 함수 ---
def log_error(error_message: str, exception: Optional[Exception] = None):
    """오류 로깅 및 화면에 표시"""
    error_details = f"{error_message}"
    if exception:
        error_details += f"\n오류 유형: {type(exception).__name__}"
        error_details += f"\n오류 내용: {str(exception)}"
    
    st.error(error_details)
    
    # 개발 모드에서만 전체 스택 트레이스 표시
    if exception and st.session_state.get("debug_mode", False):
        st.code(traceback.format_exc())

# --- 세션 상태 초기화 함수 ---
def init_session_state():
    """세션 상태 초기화 및 기본값 설정"""
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    if "model_params" not in st.session_state:
        st.session_state.model_params = {
            "temperature": 0.8,
            "top_p": 0.9,
            "repetition_penalty": 1.2,
            "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
        }
    
    if "debug_mode" not in st.session_state:
        st.session_state.debug_mode = False
    
    if "model_loaded" not in st.session_state:
        st.session_state.model_loaded = False

# --- 안전한 모델 로딩 함수 (캐싱 적용) ---
@st.cache_resource
def load_model() -> Tuple[Any, Any]:
    """모델과 토크나이저를 안전하게 로드하고 캐싱합니다."""
    try:
        with st.spinner("상담 모델을 준비하고 있어요... 잠시만 기다려주세요."):
            # 메모리 정리
            clear_cuda_cache()
            
            # 토크나이저 로드
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # 메모리 관리를 위한 모델 로딩 설정
            model_instance = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                # 학습, 저장 공간 및 성능을 위한 최적화 설정
                low_cpu_mem_usage=True,
                offload_folder="offload"
            )
            model_instance.config.use_cache = True
            model_instance.eval()
            
            st.session_state.model_loaded = True
            return tokenizer, model_instance
    except Exception as e:
        log_error("모델 로딩 중 오류가 발생했습니다:", e)
        st.session_state.model_loaded = False
        st.stop()

# --- 생성된 대화 내용을 파일로 저장하는 함수 ---
def save_conversation(messages: List[Dict[str, str]]) -> Tuple[str, io.BytesIO]:
    """대화 내용을 JSON 파일로 저장"""
    # 시스템 메시지 제외
    user_messages = [msg for msg in messages if msg["role"] != "system"]
    
    # 현재 시간을 파일명에 포함
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"conversation_{timestamp}.json"
    
    # 대화 내용을 딕셔너리로 변환
    conversation_data = {
        "timestamp": timestamp,
        "messages": user_messages
    }
    
    # JSON 파일 생성
    json_str = json.dumps(conversation_data, ensure_ascii=False, indent=2)
    json_bytes = json_str.encode('utf-8')
    buffer = io.BytesIO(json_bytes)
    
    return filename, buffer

# --- 생성된 대화 내용을 텍스트 파일로 내보내는 함수 ---
def export_conversation(messages: List[Dict[str, str]]) -> Tuple[str, io.BytesIO]:
    """대화 내용을 읽기 쉬운 텍스트 형식으로 내보내기"""
    # 시스템 메시지 제외
    user_messages = [msg for msg in messages if msg["role"] != "system"]
    
    # 현재 시간을 파일명에 포함
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"conversation_{timestamp}.txt"
    
    # 텍스트 파일 내용 생성
    content = "청소년 마음 상담 챗봇 대화 기록\n"
    content += f"시간: {datetime.datetime.now().strftime('%Y년 %m월 %d일 %H:%M:%S')}\n\n"
    
    for msg in user_messages:
        if msg["role"] == "user":
            content += f"🙋‍♀️ 내담자: {msg['content']}\n\n"
        else:
            content += f"🧑‍⚕️ 상담사: {msg['content']}\n\n"
    
    # 텍스트 파일 생성
    buffer = io.BytesIO(content.encode('utf-8'))
    
    return filename, buffer

# --- 대화 히스토리 읽기 함수 ---
def load_conversation_from_file(uploaded_file) -> bool:
    """업로드된 파일에서 대화 내용을 가져옵니다"""
    try:
        file_content = uploaded_file.read()
        loaded_data = json.loads(file_content.decode("utf-8"))
        
        # 데이터 구조 검증
        if isinstance(loaded_data, dict) and "messages" in loaded_data:
            messages = loaded_data["messages"]
            # 메시지 형식 검증
            if all(isinstance(msg, dict) and "role" in msg and "content" in msg for msg in messages):
                # 시스템 프롬프트 유지, 기존 대화 내용 교체
                st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
                return True
        
        st.error("❌ 올바른 대화 파일 형식이 아닙니다.")
        return False
    except Exception as e:
        log_error("파일을 불러오는 중 오류가 발생했습니다:", e)
        return False

# --- 안전한 모델 응답 생성 함수 ---
def generate_response(
    tokenizer: Any, 
    model: Any, 
    conversation_history: List[Dict[str, str]], 
    model_params: Dict[str, Any]
) -> str:
    """메모리 관리와 오류 처리가 강화된 응답 생성 함수"""
    try:
        # 메모리 정리
        clear_cuda_cache()
        
        # 히스토리 자르기 (메모리를 위한 안전한 길이로 제한)
        if len(conversation_history) > MAX_HISTORY_LENGTH + 1:  # +1은 시스템 프롬프트
            # 시스템 프롬프트 유지하고 최신 대화만 유지
            limited_history = [
                conversation_history[0],  # 시스템 프롬프트
                *conversation_history[-(MAX_HISTORY_LENGTH):]  # 최신 대화 내용
            ]
        else:
            limited_history = conversation_history

        # apply_chat_template 사용
        chat_prompt = tokenizer.apply_chat_template(limited_history, tokenize=False)
        chat_prompt += "<|im_start|>assistant\n"  # Assistant 시작 토큰 추가
        
        # 안전한 토큰 수로 제한
        safe_max_length = min(MAX_CONTEXT_LENGTH, tokenizer.model_max_length)
        
        # 토큰화 및 길이 제한 적용
        inputs = tokenizer(
            chat_prompt, 
            return_tensors="pt", 
            max_length=safe_max_length, 
            truncation=True
        ).to(device)

        # 응답 토큰 수 안전 제한
        safe_max_new_tokens = min(model_params["max_new_tokens"], 8192)
        
        # 토큰 생성
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=safe_max_new_tokens,
                do_sample=True,
                temperature=model_params["temperature"],
                top_p=model_params["top_p"],
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=[
                    tokenizer.convert_tokens_to_ids("<|endofturn|>"),
                    tokenizer.convert_tokens_to_ids("<|im_end|>")
                ],
                no_repeat_ngram_size=3,
                repetition_penalty=model_params["repetition_penalty"]
            )

        # 생성된 텍스트 디코딩 (입력 부분 제외)
        decoded_output = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=False)

        # 응답 추출
        response = decoded_output
        for eos in ["<|endofturn|>", "<|im_end|>", tokenizer.eos_token]:
            if eos in response:
                response = response.split(eos)[0]

        response = response.strip()

        # 비어있는 응답 방지
        if not response:
            response = "죄송해요, 답변을 생성하는 데 어려움이 있었어요. 다른 질문을 해주시겠어요?"

        # 메모리 정리
        clear_cuda_cache()
        
        return response

    except Exception as e:
        log_error("응답 생성 중 오류가 발생했습니다:", e)
        return "죄송합니다. 응답을 생성하는 데 문제가 발생했어요. 다시 시도해주시겠어요?"

# --- 세션 상태 초기화 ---
init_session_state()

# --- 모델 초기화 ---
try:
    tokenizer, model = load_model()
except Exception as e:
    log_error("모델 로딩 중 심각한 오류가 발생했습니다:", e)
    st.warning("모델 로딩에 실패했습니다. 앱을 다시 시작해주세요.")
    st.stop()

# --- 사이드바 UI 구성 ---
with st.sidebar:
    st.title("⚙️ 설정")
    st.markdown("---")
    
    # --- 파일 업로드 기능 (사이드바로 이동) ---
    st.subheader("대화 관리")
    
    # 파일 업로더 (사이드바로 이동)
    uploaded_file = st.file_uploader("📂 대화 파일 불러오기", type=["json"], key="sidebar_uploader")
    if uploaded_file is not None:
        if load_conversation_from_file(uploaded_file):
            st.success("✅ 대화를 성공적으로 불러왔어요!")
            st.rerun()
    
    # 대화 초기화 버튼
    if st.button("🗑️ 대화 초기화", key="clear_chat"):
        st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        st.success("✅ 대화가 초기화되었습니다.")
        # 메모리 정리
        clear_cuda_cache()
        st.rerun()
    
    # 대화 내용 저장/내보내기 버튼
    if len(st.session_state.messages) > 1:  # 시스템 메시지만 있는 경우 제외
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 JSON 저장"):
                filename, buffer = save_conversation(st.session_state.messages)
                st.download_button(
                    label="📥 JSON 다운로드",
                    data=buffer,
                    file_name=filename,
                    mime="application/json"
                )
        with col2:
            if st.button("📄 텍스트 내보내기"):
                filename, buffer = export_conversation(st.session_state.messages)
                st.download_button(
                    label="📥 텍스트 다운로드",
                    data=buffer,
                    file_name=filename,
                    mime="text/plain"
                )
    
    st.markdown("---")
    
    st.subheader("모델 설정")
    # 모델 매개변수 조정
    temperature = st.slider(
        "온도 (Temperature)",
        min_value=0.1,
        max_value=1.5,
        value=st.session_state.model_params["temperature"],
        step=0.1,
        help="높을수록 더 다양하고 창의적인 응답을 생성합니다. 낮을수록 더 일관되고 예측 가능한 응답을 생성합니다."
    )
    
    top_p = st.slider(
        "Top-p",
        min_value=0.1,
        max_value=1.0,
        value=st.session_state.model_params["top_p"],
        step=0.05,
        help="각 단계에서 고려할 토큰의 확률 질량입니다. 값이 작을수록 더 결정적인 응답이 됩니다."
    )
    
    repetition_penalty = st.slider(
        "반복 패널티",
        min_value=1.0,
        max_value=2.0,
        value=st.session_state.model_params["repetition_penalty"],
        step=0.1,
        help="값이 높을수록 모델이 같은 내용을 반복하지 않습니다."
    )
    
    # 안전한 범위로 제한된 토큰 수 슬라이더
    max_new_tokens = st.slider(
        "최대 토큰 수",
        min_value=128,
        max_value=8192,  # 안전한 최대값으로 제한
        value=min(DEFAULT_MAX_NEW_TOKENS, st.session_state.model_params["max_new_tokens"]),
        step=32,
        help="모델이 생성할 최대 토큰 수입니다. 값이 클수록 더 긴 응답을 생성하지만 과도한 값은 메모리 문제를 일으킬 수 있습니다."
    )
    
    # 매개변수 업데이트
    st.session_state.model_params = {
        "temperature": temperature,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "max_new_tokens": max_new_tokens
    }
    
    st.markdown("---")
    
    # 디버그 모드 토글 (개발용)
    with st.expander("🔧 고급 설정"):
        st.session_state.debug_mode = st.checkbox(
            "디버그 모드", 
            value=st.session_state.debug_mode,
            help="활성화하면 오류 발생 시 자세한 디버그 정보를 표시합니다."
        )
        
        if st.button("🧹 메모리 정리"):
            clear_cuda_cache()
            st.success("✅ 메모리를 정리했습니다.")
        
        st.caption("현재 메모리 상태:")
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            st.code(f"할당된 GPU 메모리: {allocated:.2f} GB\n예약된 GPU 메모리: {reserved:.2f} GB")
    
    # 시스템 프롬프트 표시
    with st.expander("ℹ️ 시스템 프롬프트 보기"):
        st.caption(SYSTEM_PROMPT)
    
    st.markdown("---")
    
    # 모델 상태 표시
    if st.session_state.model_loaded:
        st.success("✅ 상담 모델 로딩 완료")
    else:
        st.warning("⚠️ 모델 로딩에 문제가 있습니다")
        
    st.caption(f"📊 현재 사용 중인 기기: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        st.caption(f"💽 GPU: {torch.cuda.get_device_name(0)}")
        st.caption(f"💽 GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# --- 메인 UI 구성 ---
st.title("청소년 마음 상담 챗봇 🧑‍⚕️")
st.caption("안녕하세요! 어떤 이야기를 나누고 싶으신가요?")

# 대화 내용 표시
for i, msg in enumerate(st.session_state.messages):
    # 시스템 메시지는 표시하지 않음
    if msg["role"] != "system":
        # Streamlit의 기본 채팅 인터페이스 사용
        avatar = "🙋‍♀️" if msg["role"] == "user" else "🧑‍⚕️"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

# 사용자 입력 처리
if prompt := st.chat_input("메시지를 입력하세요..."):
    # 사용자 메시지 추가 및 표시
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🙋‍♀️"):
        st.markdown(prompt)

    # 응답 생성 중 표시
    with st.chat_message("assistant", avatar="🧑‍⚕️"):
        message_placeholder = st.empty()
        message_placeholder.markdown("생각 중...")
        
        # 응답 생성
        response = generate_response(
            tokenizer, 
            model, 
            st.session_state.messages,
            st.session_state.model_params
        )
        
        # 타이핑 효과로 응답 표시
        simulate_typing(response, message_placeholder)
    
    # 어시스턴트 응답 추가
    st.session_state.messages.append({"role": "assistant", "content": response})
