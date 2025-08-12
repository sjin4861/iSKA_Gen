# Benchmark

::: domain.entities.benchmark

## 데이터 예시

```json
[
    {
        "id": 1,
        "problem_types": [
            "제목을 붙인 근거 설명하기",
            "자문화와 비교하기",
            "원인과 전망 예측하기"
        ],
        "eval_goals": [
            "글의 전체적인 주제와 핵심 내용을 정확히 파악하는 능력을 평가한다.",
            "지문에 제시된 특정 문화 현상을 자신의 문화적 배경과 관련지어 공통점과 차이점을 구체적으로 비교 설명하는 능력을 평가한다.",
            "글에 제시된 사회/문화적 현상의 원인을 추론하고, 이를 근거로 미래에 나타날 변화나 결과를 논리적으로 설명하는 능력을 평가한다."
        ],
        "items": [
            {
                "korean_topic": "회식 문화",
                "korean_context": "회식은 한국 직장 문화의 중요한 부분으로, 업무가 끝난 후 동료들과 함께 식사하며 친목을 다지는 활동입니다. 이는 단순한 저녁 식사를 넘어 팀워크를 강화하고 조직 내 소통을 원활하게 하는 사회적 기능을 수행합니다. 하지만 최근에는 개인의 삶을 중시하는 문화가 확산되면서, 획일적이고 강압적인 회식 문화에 대한 비판과 함께 변화의 필요성이 제기되고 있습니다.",
                "foreign_topic": "Happy Hour Culture",
                "foreign_context": "Happy hour is a social tradition in many Western countries where colleagues gather at a bar or pub after work for discounted drinks. It serves as a voluntary and informal way to decompress, socialize, and build rapport outside the formal office environment. Unlike the often obligatory nature of 'Hoesik', happy hour emphasizes individual choice and casual networking."
            }
        ]
    }
]
```