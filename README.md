# iSKA-Gen: Inter-lingual Speaking Knowledge Assessment Generation

iSKA-Gen은 한국어 학습자를 위한 개인화된 말하기 평가 문항을 자동으로 생성하는 파이프라인 프로젝트입니다. 이 프로젝트는 학습자의 프로필과 관심사를 바탕으로, 두 문화권의 주제를 비교하는 고품질의 지문과 관련 문항을 생성하는 것을 목표로 합니다.

---

## 1. 프로젝트 목표

- **개인화된 문항 생성**: 학습자의 `User Profile`과 `Topic Category`를 기반으로 맞춤형 평가 문항을 생성합니다.
- **고품질 지문 확보**: `Evaluation Guideline`에 따라, 사실에 기반하고, 논리적이며, 교육적으로 적합한 비교 지문(`Passage`)을 생성합니다.
- **자동화된 평가 시스템**: 생성된 지문의 품질을 다각적으로 평가하고 우수한 지문을 선정하기 위해, 여러 `Reward Model`을 훈련하고 최적의 선정 전략을 탐색합니다.

---

## 2. 프로젝트 구조

```
/home/sjin4861/25-1/HCLT/iSKA_Gen/
├───.git/
├───.pixi/
├───.tmp/
├───saves/
├───src/
│   ├───config/
│   │   ├───prompts/
│   │   └───...
│   ├───data/
│   │   ├───rm_training/
│   │   └───...
│   ├───modules/
│   │   ├───iska/
│   │   └───...
│   ├───scripts/
│   └───utils/
├───.gitignore
├───pixi.lock
├───pixi.toml
└───README.md
```

- **`src/config`**: 모델 설정, 훈련 인자, 프롬프트 등 프로젝트의 핵심 설정 파일들을 관리합니다.
    - `prompts/iska`: 각 생성 에이전트(Topic, Context, Passage 등)가 사용하는 프롬프트 템플릿이 저장되어 있습니다.
- **`src/data`**: 훈련, 평가, 벤치마크에 사용되는 모든 데이터를 관리합니다.
    - `rm_training`: Reward Model 훈련을 위한 선호도 쌍(pairwise) 데이터셋이 저장되어 있습니다.
- **`src/modules/iska`**: `TopicAgent`, `PassageAgent` 등 실제 생성 작업을 수행하는 핵심 에이전트 모듈들이 위치합니다.
- **`src/scripts`**: 데이터셋 변환, 모델 훈련, 평가 등 주요 실험을 실행하는 스크립트들을 포함합니다.
- **`src/utils`**: 데이터 로딩, 프롬프트 관리 등 프로젝트 전반에서 사용되는 유틸리티 함수들을 제공합니다.
- **`saves/`**: 훈련된 모델의 체크포인트와 실험 결과가 저장되는 디렉토리입니다. (`.gitignore`에 포함)
- **`.tmp/`**: 임시 파일들을 저장하는 디렉토리입니다. (`.gitignore`에 포함)

---

## 3. 핵심 파이프라인 및 용어 정의

iSKA-Gen은 여러 `Agent`들이 체인 형태로 작동하여 최종 결과물을 만들어냅니다.

1.  **입력 (Top-Level Inputs)**: `User Profile`, `Topic Category`, `Evaluation Guideline`을 받습니다.
2.  **`TopicAgent`**: 비교 대상이 될 `Topic Pair` (예: 한국의 '추석'과 미국의 'Thanksgiving')를 생성합니다.
3.  **`ContextAgent`**: RAG를 통해 각 토픽에 대한 `Context` (핵심 정보 요약)를 생성합니다.
4.  **`PassageAgent`**: 두 `Context`를 바탕으로, `Evaluation Guideline`을 충족하는 비교 `Passage` (지문)를 생성합니다.
5.  **`StemAgent` & `OptionsAgent`**: 생성된 `Passage`를 기반으로 구체적인 문항(`Stem`)과 선택지(`Options`)를 만듭니다.

> 자세한 용어 정의는 [기존 용어 정의서](legacy/TERM_DEFINITION.md)를 참고하세요.

---

## 4. Reward Model 훈련 및 평가 (RM Experiment)

본 프로젝트의 핵심 과제 중 하나는 생성된 수많은 지문 후보군에서 가장 품질이 좋은 지문을 자동으로 선별하는 것입니다. 이를 위해 6가지 품질 기준(Rubric)에 따라 Reward Model(RM)을 훈련하고, 최적의 선정 방법론을 찾는 실험을 진행했습니다.

### 4.1. 실험 목표

- 6개의 개별 RM을 활용하여, 100개의 지문 후보 중 가장 우수한 **상위 25개**를 효과적으로 선정하는 최적의 방법론 탐색

### 4.2. 실험 설계

- **모델**: Qwen3 (0.6B, 1.7B, 4B)
- **데이터셋 (3종류)**:
    - **SPF (Supervised Preference Filtering)**: GPT-4o 평가를 통해 "Good" vs "Bad" 쌍 구성
    - **IMP (Inter-Model Performance)**: "고성능 모델 결과" vs "저성능 모델 결과" 쌍 구성
    - **ICP (Intra-Model Contrastive)**: 동일 모델에게 "잘 쓴 글" vs "못 쓴 글"을 의도적으로 생성시켜 쌍 구성
- **점수 합산 전략 (2가지)**:
    - **Ensemble**: 6개 RM 점수 평균
    - **Min Score**: 6개 RM 점수 중 최저점 사용
- **평가 루브릭 (6가지)**:
    1.  **평가 지침 완전성 (Completeness for Guidelines)**
    2.  **핵심 주제 명확성 (Clarity of Core Theme)**
    3.  **참고 자료 기반성 (Reference-Groundedness)**
    4.  **논리적 흐름 및 구조 (Logical Flow & Structure)**
    5.  **한국어 품질 (Korean Quality)**
    6.  **L2 학습자 적합성 (L2 Learner Suitability)**

### 4.3. 평가 방법

- 훈련된 RM을 사용하여 테스트 지문 100개의 순위를 매기고, GPT-4o 및 인간 전문가가 선정한 "Gold Standard" 상위 25개와 비교하여 **재현율(Recall@25)**과 **순위 유사도(RBO)**를 측정합니다.

> 자세한 실험 설계 및 결과는 [RM_Experiment_v1.0.0.md](RM_Experiment_v1.0.0.md) 문서를 참고하세요.

---

## 5. 실행 방법

### 스크립트 실행
모델 훈련, 평가 등 주요 작업은 `src/scripts` 내의 파이썬 스크립트를 통해 실행할 수 있습니다.

```bash
# 예시: Reward Model 훈련 스크립트 실행
python src/scripts/train_rm.py --config src/config/training_args.yaml
```

### 환경 설정
본 프로젝트는 `pixi`를 사용하여 패키지 및 환경을 관리합니다.

```bash
# 의존성 설치
pixi install

# pixi 쉘 환경 활성화
pixi shell
```