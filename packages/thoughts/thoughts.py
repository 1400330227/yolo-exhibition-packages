import os
import qianfan

# 密钥
"""
os.environ["QIANFAN_ACCESS_KEY"] =
os.environ["QIANFAN_SECRET_KEY"] = 
"""


def create_message(inputs):
    topic_list = []
    for i, x in enumerate(inputs):
        if i % 2 == 0:
            topic_list.append(f"会议主题：{x}\n")
        else:
            topic_list.append(f"会议纪要：{x}\n")
    topic_info = ""
    for x in topic_list:
        topic_info = topic_info + x

    prompt = f'''任务说明：您的任务是根据提供的会议标题和会议纪要，为用户生成一篇会议心得。这篇心得应概括会议的主要内容、关键讨论点、重要决策或行动项，并可以包含用户个人对会议的感受、学习到的知识点或对未来工作的启示。
        输入信息：{topic_info}
        输出要求：
        结构清晰：心得应包含引言、会议内容概述、关键讨论点分析、个人感受与启示等部分。
        重点突出：确保提及会议中的核心信息和重要决策，避免冗长或偏离主题的描述。
        个人视角：适当融入用户个人的思考、感受或对未来工作的建议，使心得更具个性和深度。
        语言流畅：使用清晰、准确的语言表达，确保读者能够轻松理解。
        字数控制：建议心得篇幅在300-500字之间，根据会议的重要性和复杂度可适当调整。'''
    return prompt


def do_request(inputs):
    """

    :param inputs: ["会议的主题", "会议的纪要"]
    :return:
    """
    chat_comp = qianfan.ChatCompletion()
    prompt = create_message(inputs)
    resp = chat_comp.do(model="ERNIE-4.0-8K", messages=[{"role": "user", "content": prompt}])
    return resp["body"]["result"]
