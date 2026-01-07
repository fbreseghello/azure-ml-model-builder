# Azure ML Model Builder - Documentação Técnica

## Índice

1. [Arquitetura do Sistema](#arquitetura-do-sistema)
2. [Componentes Principais](#componentes-principais)
3. [Fluxo de Dados](#fluxo-de-dados)
4. [Configuração Detalhada](#configuração-detalhada)
5. [API Reference](#api-reference)
6. [Troubleshooting](#troubleshooting)

## Arquitetura do Sistema

### Visão Geral

O Azure ML Model Builder é um sistema completo para automação de machine learning que integra:

- **Azure AutoML**: Treinamento automatizado de modelos
- **MLflow**: Rastreamento de experimentos e gerenciamento de modelos
- **Azure ML Pipelines**: Orquestração de workflows
- **Azure Container Services**: Deploy e serving de modelos

### Diagrama de Arquitetura

```
┌──────────────────────────────────────────────────────────────────┐
│                         Azure ML Workspace                        │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌────────────────┐     ┌─────────────────┐     ┌─────────────┐ │
│  │  Data Sources  │────▶│  AutoML Driver  │────▶│   Models    │ │
│  │  - Datastore   │     │  - Training     │     │  - MLflow   │ │
│  │  - Datasets    │     │  - Validation   │     │  - Registry │ │
│  └────────────────┘     └─────────────────┘     └─────────────┘ │
│                                 │                       │         │
│                                 ▼                       ▼         │
│                        ┌──────────────┐       ┌──────────────┐  │
│                        │ Explanations │       │   Endpoints  │  │
│                        │ - SHAP       │       │   - ACI      │  │
│                        │ - Feature    │       │   - AKS      │  │
│                        └──────────────┘       └──────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

## Componentes Principais

### 1. AutoML Driver (`automl_driver.py`)

**Responsabilidades:**
- Inicialização do workspace Azure ML
- Carregamento e preparação de dados
- Configuração e execução do AutoML
- Gerenciamento de experimentos

**Principais Classes/Funções:**
```python
class AutoMLDriver:
    - __init__(automl_settings, run_id)
    - run(training_percent, iteration, ...)
    - _log_configuration()
    - _try_v2_driver()
    - _run_v1_driver()
```

### 2. Scripts de Treinamento (`scripts/train_model.py`)

**Funcionalidades:**
- Interface de linha de comando para treinamento
- Configuração automática de AutoML
- Suporte a múltiplos tipos de tarefas (classificação, regressão, forecasting)
- Salvamento de métricas e artefatos

**Uso:**
```bash
python scripts/train_model.py \
    --experiment-name bike-rental \
    --task regression \
    --dataset-id <dataset-id> \
    --label-column rentals
```

### 3. Scripts de Deploy (`scripts/deploy_model.py`)

**Funcionalidades:**
- Deploy para ACI (desenvolvimento/testes)
- Deploy para AKS (produção)
- Configuração de autoscaling
- Habilitação de Application Insights

**Uso:**
```bash
python scripts/deploy_model.py \
    --model-name my-model \
    --service-name my-service \
    --deployment-target aci
```

### 4. Scripts de Teste (`scripts/test_model.py`)

**Funcionalidades:**
- Teste de endpoints deployados
- Batch testing
- Salvamento de resultados

## Fluxo de Dados

### 1. Preparação de Dados

```python
# Exemplo de carregamento de dados
from azureml.core import Dataset

dataset = Dataset.get_by_id(workspace, id='dataset-id')
df = dataset.to_pandas_dataframe()
```

### 2. Feature Engineering

O AutoML realiza automaticamente:
- Imputação de valores faltantes
- Encoding de variáveis categóricas
- Normalização/padronização
- Feature scaling
- Detecção e tratamento de outliers

### 3. Treinamento

```python
# AutoML executa múltiplos algoritmos
algorithms = [
    'RandomForest',
    'LightGBM',
    'XGBoost',
    'ElasticNet',
    # ... e outros
]

# Com diferentes hiperparâmetros
for algorithm in algorithms:
    for hyperparams in hyperparam_grid:
        train_model(algorithm, hyperparams)
        evaluate_model()
```

### 4. Seleção de Modelo

O melhor modelo é selecionado baseado na métrica primária:
- **Classificação**: AUC, Accuracy, Precision, Recall
- **Regressão**: RMSE, MAE, R²
- **Forecasting**: MAPE, RMSE

## Configuração Detalhada

### Configuração do AutoML

```python
AUTOML_SETTINGS = {
    # Controle de execução
    'experiment_timeout_minutes': 30,
    'iteration_timeout_minutes': 30,
    'max_concurrent_iterations': 4,
    'max_cores_per_iteration': -1,
    
    # Estratégias de validação
    'n_cross_validations': 5,
    'validation_size': 0.2,
    
    # Ensemble learning
    'enable_ensembling': True,
    'enable_stack_ensembling': True,
    'ensemble_iterations': 15,
    
    # Early stopping
    'enable_early_stopping': True,
    'experiment_exit_score': 0.085,
    
    # Feature engineering
    'featurization': 'auto',  # ou 'off' para desabilitar
    
    # Explicabilidade
    'model_explainability': True,
    
    # Outras opções
    'enable_onnx_compatible_models': False,
    'enable_dnn': False,
    'blacklist_algos': [],
    'whitelist_models': [],
}
```

### Configuração de Compute

```python
from azureml.core.compute import ComputeTarget, AmlCompute

compute_config = AmlCompute.provisioning_configuration(
    vm_size='STANDARD_DS11_V2',
    min_nodes=0,
    max_nodes=4,
    idle_seconds_before_scaledown=300
)
```

### Configuração de Environment

```yaml
# conda_env.yml
name: automl-env
dependencies:
  - python=3.8
  - pip:
    - azureml-defaults
    - azureml-interpret
    - scikit-learn
    - pandas
    - numpy
```

## API Reference

### AutoMLDriver

```python
class AutoMLDriver:
    """
    Main driver class for Azure AutoML training pipeline.
    
    Attributes:
        automl_settings (Dict[str, Any]): AutoML configuration
        run_id (str): Unique run identifier
        logger (logging.Logger): Logger instance
    """
    
    def __init__(self, automl_settings: Dict[str, Any], run_id: str):
        """Initialize the AutoML driver."""
        
    def run(self, 
            training_percent: int = 100,
            iteration: str = "0",
            **kwargs) -> Dict[str, Any]:
        """Execute the AutoML training pipeline."""
```

### Utility Functions

```python
def connect_to_workspace() -> Workspace:
    """Connect to Azure ML Workspace."""

def get_parent_run_id(run_id: str) -> str:
    """Extract parent run ID from child run ID."""

def initialize_directories(logger: logging.Logger) -> None:
    """Create necessary directory structure."""
```

## Troubleshooting

### Problema: Falha na Conexão com Workspace

**Sintoma:** `WorkspaceException: Unable to connect to workspace`

**Solução:**
1. Verifique as credenciais no `config.json`
2. Confirme que você tem permissões adequadas
3. Verifique a conectividade de rede

```python
# Teste de conexão
from azureml.core import Workspace
ws = Workspace.from_config()
print(f"Workspace: {ws.name}")
```

### Problema: Treinamento Muito Lento

**Possíveis Causas:**
- Compute target subdimensionado
- Dataset muito grande sem sampling
- Número excessivo de iterações

**Soluções:**
```python
# Aumentar recursos de compute
'max_concurrent_iterations': 8,

# Reduzir timeout
'experiment_timeout_minutes': 15,

# Limitar algoritmos
'whitelist_models': ['RandomForest', 'LightGBM'],
```

### Problema: Falha no Deploy

**Sintoma:** `WebserviceException: Service deployment failed`

**Verificações:**
1. Cheque logs do serviço:
   ```python
   service = Webservice(workspace, name='my-service')
   print(service.get_logs())
   ```

2. Verifique o scoring script
3. Confirme dependências no environment

### Problema: Predições Inconsistentes

**Possíveis Causas:**
- Data drift
- Dados de entrada fora da distribuição de treinamento
- Versão incorreta do modelo

**Diagnóstico:**
```python
# Verificar versão do modelo
model = Model(workspace, name='model-name')
print(f"Version: {model.version}")

# Comparar distribuição de features
compare_distributions(train_data, inference_data)
```

## Boas Práticas

### 1. Versionamento de Modelos

```python
model = Model.register(
    workspace=workspace,
    model_path='outputs/model.pkl',
    model_name='bike-rental-model',
    tags={'type': 'regression', 'framework': 'lightgbm'},
    description='Bike rental demand prediction model',
    model_framework='ScikitLearn'
)
```

### 2. Monitoramento

- Habilite Application Insights
- Configure data drift detection
- Implemente logging adequado

### 3. Segurança

- Use managed identities
- Armazene secrets no Azure Key Vault
- Habilite autenticação nos endpoints

### 4. Performance

- Use batch inferencing para grandes volumes
- Implemente caching quando apropriado
- Configure autoscaling para AKS

## Referências

- [Azure ML Documentation](https://docs.microsoft.com/azure/machine-learning/)
- [AutoML Overview](https://docs.microsoft.com/azure/machine-learning/concept-automated-ml)
- [MLflow Documentation](https://www.mlflow.org/docs/latest/index.html)
- [Scikit-learn API](https://scikit-learn.org/stable/modules/classes.html)

---

**Última Atualização:** Janeiro 2026
