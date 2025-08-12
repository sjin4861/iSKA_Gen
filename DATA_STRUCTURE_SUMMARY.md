# src/data 디렉토리 구조 및 파일 상세 설명

이 문서는 `src/data` 폴더의 구조와 각 하위 폴더 및 주요 파일의 역할을 상세히 설명하고, 실제 데이터 예시를 포함합니다.

- **`convert_dataset.py`**
  - 역할: 특정 형식의 데이터셋을 다른 형식으로 변환하는 유틸리티 스크립트입니다.

---

## `benchmarks/`
- 역할: 프로젝트의 핵심 평가 기준이 되는 벤치마크 데이터셋을 저장합니다.
- 구조: 버전별(`v1/`, `v1.1.0/`) 및 크기별(`_small`)로 관리됩니다.
- **파일 예시**: `benchmarks/v1/iSKA-Gen_Benchmark_v1.1.0_20250808_small.json`

#### 읽고 말하기 데이터 예시 (첫 번째 아이템)
```json
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
```

---

## `evaluations/`
- 역할: 모델의 성능 평가 결과를 저장합니다.
- 구조: `YYYY-MM-DD/모델명/eval_rubric/` 형태로 날짜와 모델에 따라 결과가 저장됩니다.
- **파일 예시**: `evaluations/2025-08-05/misc/A.X-4.0-Light_evaluation/eval_rubric/benchmark_1_v1.0.0_eval_rubric.json`

#### 데이터 예시 (첫 번째 아이템)
```json
{
    "id": "1_1",
    "passage": "한국의 회식 문화는 업무가 끝난 후 동료들과 식사하며 팀워크를 다지는 전통적인 사회적 기능을 수행해왔습니다. 그러나 최근 개인의 삶과 '워라밸'을 중시하는 사회적 가치관이 확산되면서, 의무적이고 획일적인 회식 문화에 대한 비판이 늘고 있습니다. 반면, 서양의 '해피아워'는 퇴근 후 동료들과 자발적으로 모여 가볍게 즐기는 비공식적인 사교 활동으로, 개인의 선택을 존중한다는 점에서 뚜렷한 차이를 보입니다. 이러한 인식 변화는 한국의 회식 문화가 과거의 수직적인 방식에서 벗어나, 점심시간을 활용하거나 문화 활동을 함께하는 등, 직원들의 자발적 참여를 유도하는 수평적이고 다양한 형태로 발전할 것임을 시사합니다.",
    "eval_rubric": {
        "clarity_of_core_theme": 5,
        "completeness_for_guidelines": 5,
        "reference_groundedness": 5,
        "logical_flow": 5,
        "korean_quality": 5,
        "l2_learner_suitability": 5
    }
}
```

---

## `pairwise_data/`
- 역할: 선호도 비교(pairwise) 데이터를 저장하며, 주로 보상 모델(Reward Model) 학습 및 평가에 사용됩니다.
- 구조: `train/`과 `test/`로 나뉘며, `train/` 내부는 `v1`, `v2`, `v3` 등 여러 버전으로 관리됩니다.

### `pairwise_data/test/`
- **파일 예시**: `pairwise_data/test/all_pairwise_data.jsonl`

#### 데이터 예시 (첫 번째 아이템)
```json
{"chosen": "한국의 회식 문화는 업무가 끝난 후 동료들과 식사하며 팀워크를 강화하는 전통적인 방식입니다. 이러한 모임은 참여가 사실상 의무적인 분위기 속에서 팀 간 소통을 촉진하는 역할을 합니다. 반면 서양의 해피 아워 문화는 할인된 음료를 마시며 자유롭게 교류하는 자발적인 행사로, 개인의 선택을 중시합니다. 한국의 회식 문화는 공동체 유대감을 강조하지만, 최근 개인의 삶을 존중하는 현대 가치관과 충돌하면서 변화가 요구되고 있으며, 앞으로 더 유연한 형태로 발전할 전망입니다.", "rejected": "한국의 회식은 a very important한 직장 문화입니다. 동료들과 저녁을 먹는것은 팀워크를 빌딩하는 좋은 방법입니다. 하지만, 그것은 가끔 강제적이고 사람들의 private life를 존중하지 안는다는 비판을 받읍니다. 서양의 해피아워는 반면에, 매우 casual하고 자발적인 모임인 것입니다. 그 차이는 매우 명확합니다."}
```

### `pairwise_data/train/`
- **파일 예시**: `pairwise_data/train/v2/icp_v1.0.0/clarity_of_core_theme/rm_pairwise.json`

#### 데이터 예시 (첫 번째 아이템)
```json
{
    "conversations": [
        {
            "from": "human",
            "value": "두 개의 한국어 독해 지문 A와 B가 주어집니다.\n두 지문은 모두 아래 [공통 조건]을 바탕으로 생성되었습니다.\n\n[공통 조건]:\n- 한국어 주제: 1인 가구 증가 현상 분석 및 사회적 시사점\n- 외국어 주제: Analysis of the Rise in Single-Person Households and its Social Implications\n\n다음은 생성된 지문의 품질을 평가하는 기준입니다:\n\n**평가 기준:**\n- 지문이 두 주제를 중심으로, 하나의 통일되고 명확한 핵심 메시지를 전달하고 있는가? 내용이 산만하거나 주제에서 벗어나는 부분이 있지는 않은가?\n\n위 기준에 따라, 두 지문 중 [핵심 주제 명확성] 측면에서 더 우수한 것을 선택하세요.\n"
        }
    ],
    "chosen": {
        "from": "gpt",
        "value": "한국 사회에서 최근 두드러진 변화 중 하나는 1인 가구의 급증입니다. 지난 10년간의 데이터를 통해 연령별로 살펴보면, 특히 청년층과 노년층에서 이러한 증가가 뚜렷하게 나타났습니다. 이는 단순한 개인 취향의 변화가 아니라 경제적 불안정성과 주거 문제, 그리고 고용 시장의 불확실성이 주요 원인으로 작용하고 있습니다. 따라서 우리는 1인 가구를 위한 맞춤형 주거 정책, 사회복지 서비스 강화, 그리고 사회적 관계망 구축의 필요성을 절실히 느끼고 있습니다. 이런 변화는 단순히 통계적 현상이 아닌, 우리 사회가 반드시 대응해야 할 중요한 과제로 다가오고 있습니다."
    },
    "rejected": {
        "from": "gpt",
        "value": "10년 전, 나는 작은 시골 마을에서 자랐고, 그때 마을 사람들은 서로를 잘 알고 있었다. 동네 구멍가게에서 마주치면 \"안녕, 영희야!\"라고 부르며 서로의 안부를 물었다. 하지만 요즘은 그런 풍경이 사라졌다. 대신 도시의 고층 아파트 단지에서 혼자 사는 사람들이 늘어나고 있다. 내가 서울의 한 오피스텔에 살고 있는 요즘, 엘리베이터에서 마주치는 이웃들은 대부분 무표정한 얼굴로 스쳐 지나간다. 가끔 배달 음식을 주문할 때면, 익명의 배달원이 문 앞에 놓고 가는 음식을 보며 고독함이 느껴졌다. 이렇게 도시에서의 삶은 개인주의적이고 단절된 느낌을 준다. 그러던 어느 날 밤, 문득 상상 속에서 나는 과거의 나를 만났다. 그는 따뜻한 공동체의 온기를 느끼며 친구들과 저녁 식사를 하고 있었다. 우리는 서로 웃고 떠들며, 삶의 소소한 행복을 나누었다. 그런데 갑자기 그가 내게 말했다. \"지금은 이런 삶이 더 나아졌다고 생각해.\" 그의 말에 잠시 당황했지만, 곧 깨달았다. 시대의 변화는 피할 수 없는 흐름이라는 것을. 그러나 여전히 내 마음 속에는 그리움과 따뜻함을 간직한 채, 새로운 형태의 공동체도 필요하다는 생각이 스며들었다."
    }
}
```

---

## `raw_outputs/`
- 역할: 모델이 생성한 원본(raw) 결과물을 저장합니다.
- 구조: `YYYY-MM-DD/생성물종류/모델명/에이전트함수명/` 형태로 매우 상세하게 나뉩니다.
- **파일 예시**: `raw_outputs/2025-08-08/passage/A.X-4.0-Light/passage_agent.create_dialogue_passage/benchmark_3_v1.1.0_passage_agent.create_dialogue_passage.json`

#### 데이터 예시 (첫 번째 아이템)
```json
{
    "id": "3_1",
    "passage": "A: 최근 사형제 존폐에 대한 논란이 다시 뜨거워지고 있는 것 같아. 너는 어떻게 생각해?\nB: 나는 반대하는 입장이야. 아무리 흉악한 범죄자라도 국가가 개인의 생명을 빼앗을 권리는 없다고 생각해. 오판의 가능성도 무시할 수 없고, 한 번 집행되면 되돌릴 수 없잖아.\nA: 일리 있는 말이야. 하지만 나는 찬성하는 입장이야. 사형제는 흉악 범죄를 예방하는 강력한 억제 효과가 있다고 생각해. 피해자와 그 가족들의 고통을 생각하면, 사회 정의를 실현하기 위해서라도 필요하다고 봐.\nB: 범죄 예방 효과에 대해서는 논란이 많아. 사형제가 없는 나라라고 해서 흉악 범죄율이 더 높다는 통계는 없잖아. 오히려 절대적 종신형처럼 다른 강력한 처벌로 대체할 수 있다고 생각해.\nA: 하지만 그건 근본적인 해결책이 될 수 없다고 봐. 사회로부터 영원히 격리한다고 해서 피해자의 상처가 치유되는 건 아니니까. 사형은 단순한 처벌을 넘어, 사회에 경종을 울리는 상징적인 의미도 있다고 생각해.\nB: 상징적인 의미 때문에 사람의 생명을 빼앗는 건 너무 위험한 발상 아닐까? 우리는 더 나은 교화 시스템을 만들고, 범죄의 근본적인 원인을 해결하는 데 집중해야 한다고 생각해. 그게 더 장기적인 관점에서 사회를 안전하게 만드는 길이야."
}
```

---

## `rm_testing/`
- 역할: 보상 모델(Reward Model)의 성능 테스트를 위한 랭킹 데이터를 저장합니다.
- **파일 예시**: `rm_testing/ranking_test_data.jsonl`

#### 데이터 예시 (첫 번째 아이템)
```json
{"passage": "한국의 '회식' 문화는 업무가 끝난 후 동료들과 식사하며 팀워크를 다지는 전통적인 사회적 기능을 수행해왔습니다. 그러나 최근 개인의 삶과 '워라밸'을 중시하는 사회적 가치관이 확산되면서, 의무적이고 획일적인 회식 문화에 대한 비판이 늘고 있습니다. 반면, 서양의 '해피아워'는 퇴근 후 동료들과 자발적으로 모여 가볍게 즐기는 비공식적인 사교 활동으로, 개인의 선택을 존중한다는 점에서 뚜렷한 차이를 보입니다. 이러한 인식 변화는 한국의 회식 문화가 과거의 수직적인 방식에서 벗어나, 점심시간을 활용하거나 문화 활동을 함께하는 등, 직원들의 자발적 참여를 유도하는 수평적이고 다양한 형태로 발전할 것임을 시사합니다.", "r1_score": "", "r2_score": "", "r3_score": "", "r4_score": "", "r5_score": "", "r6_score": "", "rm_ranking": null, "human_ranking": null}
```

---

## `rm_training/`
- 역할: 보상 모델(Reward Model) 학습용 데이터셋을 저장합니다.
- 구조: 데이터 소스(`icp`, `imp`, `spf`), 날짜, 모델 조합 등으로 복잡하게 구성됩니다.
- **파일 예시**: `rm_training/icp/2025-08-02/aidx/ICP_clarity_of_core_theme_20250802_143206.json`

#### 데이터 예시 (첫 번째 아이템)
```json
{
    "pair_id": "icp_clarity_of_core_theme_1인 가구 증가 현상 분석 및 사회적 시사점_0001",
    "rubric": "clarity_of_core_theme",
    "source_item": {
        "korean_topic": "1인 가구 증가 현상 분석 및 사회적 시사점",
        "korean_context": "오늘 저는 한국 사회의 가장 큰 변화 중 하나인 '1인 가구의 증가' 현상에 대해 발표하겠습니다. 먼저 지난 10년간 1인 가구의 증가 추이를 연령대별 그래프로 보여드리고, 그 원인을 사회경제적 측면에서 분석하겠습니다. 특히 청년층과 노년층에서 1인 가구가 급증하고 있으며, 이는 개인의 가치관 변화와 함께 주거 및 고용 불안의 심화를 의미합니다. 결론적으로, 우리 사회는 이제 1인 가구를 위한 맞춤형 주거, 복지, 그리고 사회적 관계망 지원 정책을 시급히 마련해야 합니다.",
        "foreign_topic": "Analysis of the Rise in Single-Person Households and its Social Implications",
        "foreign_context": "Today, I will discuss one of Korea's most significant social changes: the increase in single-person households. I will first show the trend over the past 10 years with a graph broken down by age, and then analyze the socioeconomic causes. The sharp increase, particularly among the young and elderly, reflects not only changing values but also growing housing and employment instability. In conclusion, our society must urgently develop policies for housing, welfare, and social networks tailored to this demographic.",
        "problem_types": [
            "발표의 핵심 목적 파악하기",
            "세부 내용 및 근거 설명하기",
            "전체 내용 요약 및 재구성하기"
        ],
        "eval_goals": [
            "주어진 발표문(지문)의 전체적인 주제와 핵심 목적이 무엇인지 정확히 파악하는 능력을 평가한다.",
            "발표의 주장을 뒷받침하기 위해 사용된 데이터와 근거를 파악하고, 그 내용을 상세히 설명하는 능력을 평가한다.",
            "발표 전체 내용을 자신의 말로 간결하게 요약하고, 핵심 메시지를 재구성하여 전달하는 능력을 평가한다."
        ]
    },
    "chosen": "1인 가구의 증가 현상에 대한 발표에서는, 주로 주택과 관련된 사회적 이슈들을 다루었다. 먼저, 최근 10년간의 1인 가구 증가율을 보여주는 그래프를 통해 이 현상을 시각화하였으며, 이는 청년층의 고용 불안정과 노인의 고독감과 관련이 있다. 이러한 경향은 개인의 가치관 변화뿐만 아니라 주거와 경제적 환경에서도 영향을 미친다고 볼 수 있다. 따라서 우리는 이 문제를 해결하기 위해서는 맞춤형 주거 지원과 사회적 네트워크 강화 정책이 필요하다.  지문 길이: 약 250자",
    "rejected": "요즘 집에 혼자 있는 날들이 많아서 내가 만든 집에서 시간을 보내는 것이 정말 편리하다는 생각이 든다. 내가 좋아하는 음식을 많이 만들어 먹을 수 있고, TV 프로그램이나 영화를 마음껏 볼 수 있으니 정말 즐겁다. 하지만 때때로 혼자 있는 것만큼도 외롭게 느껴질 때가 있다. 그러므로 이런 경험들은 나에게 많은 감정을 안겨준다. 최근에는 한 명의 조카가 생기면서 새로운 가족이 생겼고, 그럴 때마다 소소한 일상을 공유하며 행복해진다. 물론 이 모든 이야기는 주제와는 거리가 멉니다. 하지만 혼자 사는 것과 같은 삶은 어떤 방식으로든 중요한 부분을 차지할 수 있다는 것을 깨닫게 해줍니다.",
    "dataset_type": "ICP",
    "created_at": "2025-08-02T14:32:06.480355"
}
```
```