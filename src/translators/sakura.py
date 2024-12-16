import httpx

class SakuraTranslator:
    def __init__(self, api_url: str, timeout: int = 60):
        self.api_url = api_url
        self.timeout = timeout

    def generate_prompt(self, japanese, gpt_dict):
        gpt_dict_text_list = []
        for gpt in gpt_dict:
            src = gpt['src']
            dst = gpt['dst']
            info = gpt.get('info', None)
            if info:
                single = f"{src}->{dst} #{info}"
            else:
                single = f"{src}->{dst}"
            gpt_dict_text_list.append(single)

        if gpt_dict_text_list:
            gpt_dict_raw_text = "\n".join(gpt_dict_text_list)
            user_prompt = (
                "根据以下术语表（可以为空）：\n"
                + gpt_dict_raw_text
                + "\n将下面的日文文本根据对应关系和备注翻译成中文："
                + japanese
            )
        else:
            user_prompt = "将下面的日文文本翻译成中文：" + japanese

        prompt = (
            "<|im_start|>system\n"
            "你是一个轻小说翻译模型，可以流畅通顺地以日本轻小说的风格将日文翻译成简体中文，"
            "并联系上下文正确使用人称代词，不擅自添加原文中没有的代词。<|im_end|>\n"
            + "<|im_start|>user\n"
            + user_prompt
            + "<|im_end|>\n"
            + "<|im_start|>assistant\n"
        )
        return prompt

    def translate(self, japanese, gpt_dict=[]):
        prompt = self.generate_prompt(japanese, gpt_dict)

        payload = {
            "model": "gpt-3.5-turbo",  # 模型名称，根据你的接口配置调整
            "messages": [
                {"role": "system", "content": "你是一个轻小说翻译模型。"},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,  # 可调整，控制生成文本的随机性
        }
        try:
            response = httpx.post(self.api_url, json=payload, timeout=self.timeout)
            if response.status_code == 200:
                response_data = response.json()
                # 假设返回格式为 {"choices": [{"message": {"content": "翻译结果"}}]}
                return response_data["choices"][0]["message"]["content"].strip()
            else:
                response.raise_for_status()
        except httpx.RequestError as e:
            return f"请求失败：{e}"

# 示例使用
if __name__ == "__main__":
    # 定义 gpt_dict 和输入文本
    gpt_dict = [
        {"src": "原文1", "dst": "译文1", "info": "注释信息1"},
        {"src": "原文2", "dst": "译文2"},
    ]
    japanese_text = """．．．
．．．
救ったのは、土河！
カナち！！
．．．０円スマイル？
¢岩出雪ネま林粉／失敗注マ少ミ工梨快愛飼絵
上空にあらかじめ配置を．．．！？
原作：岩田話待ってた原作・岩田雪花・作画・青木裕
サービスのお客様からご利用いただきましたが、お客さまにお気になりますのですが、"""

    # llama.cpp 接口地址
    api_url = "http://192.168.100.113:8080/v1/chat/completions"

    # 创建翻译器对象
    translator = SakuraTranslator(api_url)

    # 调用翻译方法
    result = translator.translate(japanese_text, gpt_dict)
    print("翻译结果：", result)
