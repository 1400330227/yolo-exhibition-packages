context = {
    'person': '人',
    'bicycle': '自行车',
    'car': '汽车',
    'motorcycle': '摩托车',
    'airplane': '飞机',
    'bus': '公交车',
    'train': '火车',
    'fire,': '火焰',
    'smoke,': '烟雾',
    '_backg': '背景',
    'backg': '背景',
    'other,': '似火焰',
    'head': '头部',
    'helmet': '头盔',
    'cra,': '龟裂',
    'inc,': '杂质',
    'pat,': '斑块',
    'pit,': '点蚀',
    'rol,': '氧化层',
    'scr': '划痕',
    'black': '黑色',
    'blue': '蓝色',
    'green': '绿色',
    'red': '红色',
    'white': '白色',
}


def i18n(value):
    if str(value) in context:
        return context[value]
    else:
        return value
