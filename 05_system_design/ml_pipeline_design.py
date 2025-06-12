"""
🎯 문제: ML 파이프라인 설계

확장 가능하고 유지보수가 용이한 머신러닝 파이프라인을 설계하고 구현하세요.

요구사항:
1. 데이터 수집 → 전처리 → 학습 → 평가 → 배포 파이프라인
2. 설정 기반 파이프라인 (YAML/JSON)
3. 로깅 및 모니터링
4. 실패 처리 및 재시도 로직
5. 모델 버전 관리
6. A/B 테스트 지원

아키텍처 고려사항:
- 모듈화 및 재사용성
- 확장성 (스케일링)
- 장애 복구
- 데이터 품질 검증
"""

import os
import json
import yaml
import logging
import pickle
from datetime import datetime
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

@dataclass
class PipelineConfig:
    """
    파이프라인 설정 클래스
    """
    # TODO: 설정 클래스 구현
    pass

class PipelineStep(ABC):
    """
    파이프라인 스텝 추상 클래스
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.logger = logging.getLogger(f"pipeline.{name}")
    
    @abstractmethod
    def execute(self, data: Any) -> Any:
        """
        스텝 실행
        
        Args:
            data: 입력 데이터
            
        Returns:
            processed_data: 처리된 데이터
        """
        pass
    
    def validate_input(self, data: Any) -> bool:
        """
        입력 데이터 검증
        
        Args:
            data: 검증할 데이터
            
        Returns:
            is_valid: 유효성 여부
        """
        # TODO: 입력 검증 구현
        return True
    
    def log_metrics(self, metrics: Dict[str, Any]):
        """
        메트릭 로깅
        
        Args:
            metrics: 로깅할 메트릭
        """
        self.logger.info(f"Step {self.name} metrics: {metrics}")

class DataIngestionStep(PipelineStep):
    """
    데이터 수집 스텝
    """
    
    def execute(self, data_source: str) -> pd.DataFrame:
        """
        데이터 수집 실행
        
        Args:
            data_source: 데이터 소스 경로
            
        Returns:
            raw_data: 수집된 원시 데이터
        """
        # TODO: 데이터 수집 구현
        pass

class DataPreprocessingStep(PipelineStep):
    """
    데이터 전처리 스텝
    """
    
    def execute(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        데이터 전처리 실행
        
        Args:
            raw_data: 원시 데이터
            
        Returns:
            processed_data: 전처리된 데이터
        """
        # TODO: 데이터 전처리 구현
        pass

class ModelTrainingStep(PipelineStep):
    """
    모델 학습 스텝
    """
    
    def execute(self, processed_data: pd.DataFrame) -> Any:
        """
        모델 학습 실행
        
        Args:
            processed_data: 전처리된 데이터
            
        Returns:
            trained_model: 학습된 모델
        """
        # TODO: 모델 학습 구현
        pass

class ModelEvaluationStep(PipelineStep):
    """
    모델 평가 스텝
    """
    
    def execute(self, model_and_data: tuple) -> Dict[str, float]:
        """
        모델 평가 실행
        
        Args:
            model_and_data: (모델, 테스트 데이터) 튜플
            
        Returns:
            evaluation_metrics: 평가 메트릭
        """
        # TODO: 모델 평가 구현
        pass

class ModelDeploymentStep(PipelineStep):
    """
    모델 배포 스텝
    """
    
    def execute(self, model_and_metrics: tuple) -> str:
        """
        모델 배포 실행
        
        Args:
            model_and_metrics: (모델, 메트릭) 튜플
            
        Returns:
            deployment_info: 배포 정보
        """
        # TODO: 모델 배포 구현
        pass

class MLPipeline:
    """
    머신러닝 파이프라인 메인 클래스
    """
    
    def __init__(self, config_path: str):
        """
        파이프라인 초기화
        
        Args:
            config_path: 설정 파일 경로
        """
        # TODO: 파이프라인 초기화 구현
        pass
    
    def _load_config(self, config_path: str) -> PipelineConfig:
        """
        설정 파일 로드
        
        Args:
            config_path: 설정 파일 경로
            
        Returns:
            config: 파이프라인 설정
        """
        # TODO: 설정 로드 구현
        pass
    
    def _setup_logging(self):
        """
        로깅 설정
        """
        # TODO: 로깅 설정 구현
        pass
    
    def _create_steps(self) -> List[PipelineStep]:
        """
        파이프라인 스텝 생성
        
        Returns:
            steps: 파이프라인 스텝 리스트
        """
        # TODO: 스텝 생성 구현
        pass
    
    def run(self, data_source: str) -> Dict[str, Any]:
        """
        파이프라인 실행
        
        Args:
            data_source: 데이터 소스
            
        Returns:
            results: 파이프라인 실행 결과
        """
        # TODO: 파이프라인 실행 구현
        pass
    
    def _handle_failure(self, step: PipelineStep, error: Exception, retry_count: int = 0):
        """
        실패 처리 및 재시도
        
        Args:
            step: 실패한 스텝
            error: 발생한 에러
            retry_count: 재시도 횟수
        """
        # TODO: 실패 처리 구현
        pass
    
    def save_model_version(self, model: Any, version: str, metrics: Dict[str, float]):
        """
        모델 버전 저장
        
        Args:
            model: 저장할 모델
            version: 모델 버전
            metrics: 모델 성능 메트릭
        """
        # TODO: 모델 버전 관리 구현
        pass
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        파이프라인 상태 조회
        
        Returns:
            status: 파이프라인 상태 정보
        """
        # TODO: 상태 조회 구현
        pass

# 테스트 코드
if __name__ == "__main__":
    # 설정 파일 생성 (예시)
    config = {
        "pipeline": {
            "name": "ml_pipeline_example",
            "version": "1.0.0",
            "steps": [
                {
                    "name": "data_ingestion",
                    "type": "DataIngestionStep",
                    "config": {"source_type": "csv"}
                },
                {
                    "name": "data_preprocessing",
                    "type": "DataPreprocessingStep",
                    "config": {"scaling": "standard", "encoding": "onehot"}
                },
                {
                    "name": "model_training",
                    "type": "ModelTrainingStep",
                    "config": {"algorithm": "random_forest", "n_estimators": 100}
                },
                {
                    "name": "model_evaluation",
                    "type": "ModelEvaluationStep",
                    "config": {"metrics": ["accuracy", "precision", "recall"]}
                },
                {
                    "name": "model_deployment",
                    "type": "ModelDeploymentStep",
                    "config": {"deployment_type": "api", "threshold": 0.85}
                }
            ]
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        },
        "retry": {
            "max_attempts": 3,
            "delay": 5
        }
    }
    
    # 설정 파일 저장
    with open("pipeline_config.yaml", "w") as f:
        yaml.dump(config, f)
    
    # 파이프라인 실행
    pipeline = MLPipeline("pipeline_config.yaml")
    
    # 샘플 데이터 소스
    data_source = "sample_data.csv"
    
    try:
        results = pipeline.run(data_source)
        print("파이프라인 실행 완료:")
        print(json.dumps(results, indent=2))
        
        # 파이프라인 상태 확인
        status = pipeline.get_pipeline_status()
        print("\n파이프라인 상태:")
        print(json.dumps(status, indent=2))
        
    except Exception as e:
        print(f"파이프라인 실행 실패: {e}")
    
    # 정리
    if os.path.exists("pipeline_config.yaml"):
        os.remove("pipeline_config.yaml")