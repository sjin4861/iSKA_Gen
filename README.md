# iSKA-Gen 프로젝트 용어 정의서

이 문서는 iSKA-Gen 프로젝트의 논문과 코드 베이스에서 사용되는 핵심 용어들을 정의하여, 팀원 간의 명확한 소통과 일관성 있는 문서화를 돕기 위해 작성되었습니다.

---

## 1. 최상위 입력 (Top-Level Inputs)

파이프라인을 시작하는 데 사용되는 초기 입력 데이터입니다.

### User Profile
-   **정의**: 한국어 학습자의 인구통계학적 정보, 학습 동기, 현재 언어 수준, 관심사 등을 구조화한 데이터입니다.
-   **역할**: 개인화된 `Topic` 추천의 기반이 됩니다.
-   **코드**: `Dict[str, Any]`

### Topic Category
-   **정의**: "명절", "음식", "사회 문제"와 같이, 생성할 `Topic Pair`의 상위 주제 분류입니다.
-   **역할**: `TopicAgent`가 특정 범위 내에서 토픽을 탐색하도록 제한하는 역할을 합니다.
-   **코드**: `str`

### Evaluation Guideline (평가 지침)
-   **정의**: 최종적으로 생성될 말하기 평가 문항이 어떤 능력을 측정해야 하는지 정의한 `<문제 유형, 평가 목표>` 쌍의 집합입니다. 보통 3개의 쌍으로 구성됩니다.
-   **역할**: 파이프라인의 모든 생성 단계(토픽, 컨텍스트, 지문 등)에 일관된 방향성을 제시하는 핵심적인 지침입니다.
-   **코드**: `List[Dict[str, str]]`

---

## 2. 생성 에이전트 (Generative Agents)

각각의 특정 생성 작업을 수행하는 독립적인 LLM 기반 모듈입니다.

### TopicAgent
-   **역할**: `User Profile`, `Topic Category`, `Evaluation Guideline`을 입력받아, 비교에 적합하면서도 위키피디아에서 검색 가능한 `Topic Pair`를 생성합니다.

### ContextAgent
-   **역할**: `Topic Pair`와 `Evaluation Guideline`을 입력받아, RAG(검색 증강 생성)를 통해 각 토픽에 대한 `home_context`와 `foreign_context`를 생성합니다.

### PassageAgent
-   **역할**: `home_context`, `foreign_context`, `Evaluation Guideline`을 입력받아, 모든 평가 목표를 충족하는 최종 `Passage`(비교 지문)를 생성합니다.

### StemAgent
-   **역할**: 생성된 `Passage`와 개별 `Evaluation Guideline`을 입력받아, 학습자에게 제시될 구체적인 문항(`Stem`)을 생성합니다.

### OptionsAgent
-   **역할**: 4지선다형이 필요한 `Stem`에 대해, 하나의 정답과 세 개의 매력적인 오답으로 구성된 `Options`를 생성합니다.

---

## 3. 핵심 데이터 엔티티 (Core Data Entities)

파이프라인의 각 단계를 거치며 생성되고 전달되는 데이터 조각들입니다.

### Topic Pair
-   **정의**: 비교 대상이 되는 두 문화 주제의 쌍입니다.
    -   `home_topic`: 한국 문화 주제 (한국어)
    -   `foreign_topic`: 비교 대상이 되는 외국 문화 주제 (영어)
-   **역할**: `ContextAgent`의 핵심 입력값입니다.

### Context
-   **정의**: RAG를 통해 수집된 정보를, `Evaluation Guideline`에 맞춰 2~3 문장으로 요약한 핵심 내용입니다.
    -   `home_context`: `home_topic`에 대한 요약
    -   `foreign_context`: `foreign_topic`에 대한 요약
-   **역할**: `PassageAgent`가 최종 지문을 생성하는 데 사용하는 "재료"입니다.

### Passage (지문)
-   **정의**: `home_context`와 `foreign_context`를 바탕으로, 여러 `Evaluation Guideline`을 모두 충족하도록 생성된 하나의 통합된 비교 설명글입니다.
-   **역할**: `StemAgent`와 `OptionsAgent`가 최종 문항을 만드는 기반이 되는 자극(stimulus)입니다.

### Stem (문항)
-   **정의**: 선택지를 제외한, 문제의 핵심적인 질문이나 과제를 제시하는 텍스트입니다.
-   **역할**: 학습자가 무엇을 해야 하는지 명확하게 지시합니다.

### Options (선택지)
-   **정의**: 4지선다형 문항을 위해 생성된, 하나의 정답과 세 개의 오답으로 구성된 리스트입니다.
-   **역할**: 객관식 평가를 가능하게 합니다.

---

## 4. 평가 지표 (Evaluation Rubrics)

생성된 `Passage`의 품질을 O/X로 평가하는 데 사용되는 기준입니다.

-   **사실 기반성 (Factual Consistency)**: 지문이 참고 자료의 내용과 사실적으로 일치하는가?
-   **연결 자연스러움 (Naturalness)**: 두 주제가 하나의 글로 자연스럽게 연결되었는가?
-   **한국어 품질 (Korean Quality)**: 문법, 어휘, 문장 구조가 자연스러운가?
-   **L2 학습자 적합성 (L2 Learner Suitability)**: 읽는 난이도가 외국인 학습자에게 적절한가?


1.  **평가 지침 부합성 (Guideline Compliance):** 생성된 지문의 내용이, 주어진 '평가 지침'에 명시된 3가지 목표를 달성하는 데 필요한 정보를 충분히 포함하고 있는가?
    2.  **사실 기반성 (Factual Consistency):** 생성된 지문의 모든 내용이 함께 제공된 '참고 자료'에 의해서만 뒷받침되는가? (참고 자료에 없는 내용이 포함되면 'false')
    3.  **연결 자연스러움 (Naturalness):** '한국 문화'와 '외국 문화'라는 두 주제가 단순히 나열된 것이 아니라, 하나의 통일된 글로 자연스럽게 연결되었는가?
    4.  **한국어 품질 (Korean Quality):** 지문에 사용된 어휘와 문장 구조가 문법적으로 정확하고, 외국인 한국어 학습자가 이해하기에 자연스러운가? (어색하거나 번역투의 표현이 없는가?)
    5.  **L2 학습자 적합성 (L2 Learner Suitability):** 지문의 전반적인 난이도가 외국인 한국어 학습자에게 적절한가? (문맥상 추론하기 어려운 전문 용어나 한자어가 별도의 설명 없이 사용되지는 않았는가?)
