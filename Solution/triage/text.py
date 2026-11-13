"""Conservative matching views; original text is never rewritten."""
import re
import unicodedata

# Small, explicit character map; no language-model or dictionary download needed.
_PAIRS = "压壓 机機 电電 热熱 温溫 烫燙 声聲 响響 动動 转轉 风風 没沒 关關 开開 烧燒 线線 断斷 来來 还還 问問 题題 坏壞 这這 边邊 个個 为為 么麼 过過 时時 间間 进進 无無 变變 频頻 数數 值值 图圖 连連 网網 络絡 讯訊 号號 误誤 码碼 软軟 载載 复復 启啟 发發 现現 后後 门門 仓倉 库庫 顶頂 楼樓 泵泵 输輸 带帶 马馬 达達 净淨 湿濕 溼濕 灯燈 跳跳 脱脫 积積 油油 触觸 倒倒 裂裂 毒毒 异異 味味 漏漏 听聽 说說 应應 该該 乎乎 晓曉 得得 帮幫 钮鈕 键鍵 画畫 面面 扫掃 条條 读讀 迟遲 迟遲 滤濾 脏髒 锈鏽 壳殼 松鬆 丝絲 报報 修修 锁鎖 齿齒 厂廠 实實 冷冷 备備 护護 龟龜 尘塵 严嚴 齐齊 震震 预預 满滿 确確 认認 应應 警警 静靜 粘黏 阀閥 噪噪 循循 环環 冻凍 拨撥 拖拖".split()
_MAP = str.maketrans({p[0]: p[1] for p in _PAIRS})


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(_MAP).lower()
    text = re.sub(r"[\u200b-\u200d\ufeff]", "", text)
    # Collapse repeated CJK characters, but preserve identifiers and error-code digits.
    return re.sub(r"([\u3400-\u9fff])\1+", r"\1", text)


def urgency_text(text: str) -> str:
    return normalize(text)  # No error-code meaning, synonyms or previous text added.


def classification_text(text: str) -> str:
    text = normalize(text)
    return re.split(r"\n\s*--\s*\n|從我的\s*iphone\s*傳送", text)[0]


def matches(term: str, text: str, *, missing: bool = False) -> bool:
    term = normalize(term)
    compact = re.sub(r"\s+", "", text)
    target = re.sub(r"\s+", "", term)
    if target in compact:
        return True
    if not missing or len(target) < 3:
        return False
    # One missing character only; require the remaining characters as evidence.
    for i in range(len(target)):
        pattern = re.escape(target[:i]) + r"(?:□|�|\?{1,2})" + re.escape(target[i + 1:])
        if re.search(pattern, compact):
            return True
    return False


def error_codes(text: str) -> list[str]:
    return list(dict.fromkeys(f"E-{int(n):02d}" for n in
                             re.findall(r"(?<![a-z0-9])e[\s_-]*(\d{1,2})(?!\d)", normalize(text))))
