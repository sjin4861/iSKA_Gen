# 🔧 Generator 리팩토링 TODO 리스트

## 📅 **프로젝트 개요**
- **시작일**: 2025-08-10
- **목표**: ArgParser 기반 구조화된 생성 시스템 구축
- **현재 상태**: 계획 수립 완료, 구현 시작

---

## ✅ **완료된 작업**
- [x] 리팩토링 계획 수립
- [x] TODO 리스트 작성
- [x] **ArgumentParser 구현**
  - [x] 6개 필수 파라미터 정의
  - [x] 서브타입 처리 (Material의 passage/audio_script/image_caption/all)
  - [x] 모델 리스트 파싱
  - [x] 날짜 형식 검증
  - [x] Stem 생성용 추가 옵션 (passage-models, passage-date)
- [x] **ConfigManager 구현**
  - [x] 벤치마크 매핑 설정 파일 생성
  - [x] 동적 타입-ID 매핑 로직
  - [x] 최신 벤치마크 버전 관리
- [x] **기본 구조 설계**
  - [x] BaseGenerator 추상 클래스 구현
  - [x] MaterialGenerator 구현
  - [x] StemGenerator 구현
  - [x] 새로운 메인 generator_new.py 구현
  - [x] 디렉토리 구조 생성 및 __init__.py 파일 추가
- [x] **OutputManager 구현**
  - [x] JSON 파일 저장/로드 기능
  - [x] 백업 및 복구 기능
  - [x] NULL 항목 관리
  - [x] 생성 리포트 기능

---

## 🚧 **진행 중인 작업**

### **Phase 2: 핵심 기능 구현** ✅ **대부분 완료**
- [x] **MaterialGenerator 구현**
- [x] **StemGenerator 리팩토링**
  - [x] 기존 3개 동시 생성 로직 유지
  - [x] 지문-Stem 모델 조합 처리
  - [x] 메모리 관리 최적화
- [x] **OutputManager 구현**
- [ ] **통합 테스트 및 검증** 🚧 **진행 중**

---

## 📋 **대기 중인 작업**

### **Phase 3: 고급 기능 및 최적화**
- [ ] **자동화 로직 구현**
  - [ ] 평가 지침 자동 결정
  - [ ] Source Item 자동 추출
  - [ ] 벤치마크 정보 자동 매핑

### **Phase 4: 완성도 향상**
- [ ] **OutputManager 구현**
  - [ ] 결과 저장 경로 통합 관리
  - [ ] 백업 및 복구 기능
  - [ ] 진행 상황 로깅
  
- [ ] **에러 처리 및 검증**
  - [ ] 입력 파라미터 검증
  - [ ] 모델 존재 여부 확인
  - [ ] 벤치마크 파일 검증
  
- [ ] **성능 최적화**
  - [ ] 메모리 사용량 최적화
  - [ ] 병렬 처리 개선
  - [ ] 중간 결과 캐싱

---

## 🎯 **주요 설계 원칙**

### **유연성**
- 하드코딩 제거, 설정 기반 동적 매핑
- 벤치마크 버전 변경에 대한 자동 대응
- 모델 리스트 자유 조합

### **모듈화**
- 기능별 독립적인 클래스/모듈 분리
- 재사용 가능한 컴포넌트 설계
- 테스트 용이한 구조

### **사용성**
- 직관적인 CLI 인터페이스
- 자동 설정과 수동 설정의 균형
- 명확한 에러 메시지

---

## 📁 **목표 파일 구조**

```
src/scripts/
├── generator.py              # 메인 CLI 진입점
├── config/
│   ├── benchmark_mapping.yaml  # 벤치마크 타입 매핑
│   └── model_config.yaml      # 모델 설정
├── generators/
│   ├── __init__.py
│   ├── base_generator.py      # 베이스 클래스
│   ├── material_generator.py  # Material 생성기
│   ├── stem_generator.py      # Stem 생성기
│   └── source_item_generator.py # 향후 확장
├── managers/
│   ├── __init__.py
│   ├── config_manager.py      # 설정 관리
│   ├── output_manager.py      # 출력 관리
│   └── validation_manager.py  # 검증 관리
└── utils/
    ├── __init__.py
    ├── argument_parser.py     # CLI 파싱
    └── benchmark_utils.py     # 벤치마크 유틸리티
```

---

## 🚀 **사용 예시 (목표)**

### **기본 사용법**
```bash
# Passage 생성 (자동 설정)
python generator.py --type Material --subtype passage --date 2025-08-10

# 특정 모델로 Stem 생성  
python generator.py --type Stem --models EXAONE-3.5-7.8B-Instruct --date 2025-08-10

# 모든 Material 생성
python generator.py --type Material --subtype all --date 2025-08-10
```

### **고급 사용법**
```bash
# 수동 지정으로 생성
python generator.py --type Material --subtype image_caption \
  --criteria v1.1.0:5:0 --source v1.1.0:5:10 --date 2025-08-10

# 다중 모델로 생성
python generator.py --type Material --subtype passage \
  --models "EXAONE-3.5-7.8B-Instruct,Midm-2.0-Base-Instruct" --date 2025-08-10
```

---

## 📊 **진행 상황 추적**

- **전체 진행률**: 90% (18/20 작업 완료)
- **현재 단계**: Phase 2 완료, 통합 테스트 진행 중
- **다음 작업**: 실제 환경에서 종합 테스트
- **예상 완료일**: 2025-08-10 (당일 완료 가능)

---

## 📝 **작업 로그**

### **2025-08-10**
- ✅ 리팩토링 계획 수립 완료
- ✅ TODO 리스트 마크다운 파일 생성
- ✅ ArgumentParser 구현 완료 (추가 옵션 포함)
- ✅ ConfigManager 구현 완료
- ✅ BaseGenerator 추상 클래스 구현
- ✅ MaterialGenerator 구현 완료
- ✅ StemGenerator 구현 완료
- ✅ OutputManager 구현 완료
- ✅ 새로운 메인 CLI 인터페이스 (generator_new.py) 구현
- ✅ 디렉토리 구조 및 설정 파일 생성
- ✅ 모든 핵심 컴포넌트 구현 완료
- 🚧 통합 테스트 및 최종 검증 준비

---

## ⚠️ **주의사항 및 제약조건**

1. **기존 기능 호환성**: 기존 생성 결과와 동일한 품질 유지
2. **메모리 관리**: GPU 메모리 사용량 최적화 필수
3. **에러 복구**: 부분 실패시 복구 가능한 구조
4. **확장성**: Source Item 생성 기능 추후 추가 고려

---

*마지막 업데이트: 2025-08-10*
