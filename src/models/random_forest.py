import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import joblib
import mlflow
import mlflow.sklearn
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import logging


# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChurnRandomForestModel:
    """
    Modelo Random Forest para predição de churn de clientes
    
    Este modelo utiliza Random Forest com otimização de hiperparâmetros
    para predizer a probabilidade de churn de clientes com alta acurácia.
    """
    
    def __init__(self, random_state: int = 42):
        self.model = None
        self.random_state = random_state
        self.feature_importance = None
        self.best_params = None
        self.training_metrics = {}
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              hyperparameter_tuning: bool = True) -> Dict:
        """
        Treina o modelo Random Forest com opção de otimização de hiperparâmetros
        
        Args:
            X_train: Features de treinamento
            y_train: Target de treinamento  
            hyperparameter_tuning: Se deve fazer tuning de hiperparâmetros
            
        Returns:
            Dict com métricas de treinamento
        """
        
        logger.info("🚀 Iniciando treinamento do modelo Random Forest")
        
        with mlflow.start_run(run_name="RandomForest_Churn_Prediction"):
            
            if hyperparameter_tuning:
                logger.info("🔧 Executando otimização de hiperparâmetros...")
                
                # Grid de hiperparâmetros otimizado
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, 30, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None],
                    'bootstrap': [True, False],
                    'class_weight': ['balanced', None]
                }
                
                rf = RandomForestClassifier(
                    random_state=self.random_state, 
                    n_jobs=-1,
                    criterion='gini'
                )
                
                # Grid Search com validação cruzada
                grid_search = GridSearchCV(
                    estimator=rf,
                    param_grid=param_grid,
                    cv=5,
                    scoring='roc_auc',
                    n_jobs=-1,
                    verbose=1,
                    return_train_score=True
                )
                
                grid_search.fit(X_train, y_train)
                self.model = grid_search.best_estimator_
                self.best_params = grid_search.best_params_
                
                # Log parâmetros otimizados
                mlflow.log_params(self.best_params)
                mlflow.log_metric("best_cv_score", grid_search.best_score_)
                
                logger.info(f"✅ Melhores parâmetros encontrados: {self.best_params}")
                
            else:
                # Usar parâmetros padrão otimizados baseados em experiência
                self.model = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=20,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    max_features='sqrt',
                    bootstrap=True,
                    class_weight='balanced',
                    random_state=self.random_state,
                    n_jobs=-1
                )
                
                logger.info("🔧 Usando parâmetros padrão otimizados")
                self.model.fit(X_train, y_train)
            
            # Validação cruzada para métricas robustas
            logger.info("📊 Executando validação cruzada...")
            cv_scores = cross_val_score(
                self.model, X_train, y_train, 
                cv=5, scoring='roc_auc', n_jobs=-1
            )
            
            # Calcular feature importance
            self.feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Métricas de treinamento
            self.training_metrics = {
                'cv_mean_auc': cv_scores.mean(),
                'cv_std_auc': cv_scores.std(),
                'feature_count': len(X_train.columns),
                'training_samples': len(X_train)
            }
            
            # Log métricas no MLflow
            for metric, value in self.training_metrics.items():
                mlflow.log_metric(metric, value)
            
            # Log modelo
            mlflow.sklearn.log_model(
                self.model, 
                "random_forest_model",
                registered_model_name="ChurnPredictionRandomForest"
            )
            
            # Log feature importance como artifact
            importance_path = "feature_importance.csv"
            self.feature_importance.to_csv(importance_path, index=False)
            mlflow.log_artifact(importance_path)
            
            logger.info(f"✅ Modelo treinado com sucesso!")
            logger.info(f"📊 AUC médio: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            return self.training_metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Faz predições binárias (0/1)"""
        if self.model is None:
            raise ValueError("❌ Modelo não foi treinado ainda. Execute .train() primeiro.")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Retorna probabilidades de predição [prob_classe_0, prob_classe_1]"""
        if self.model is None:
            raise ValueError("❌ Modelo não foi treinado ainda. Execute .train() primeiro.")
        
        return self.model.predict_proba(X)
    
    def get_churn_probability(self, X: pd.DataFrame) -> np.ndarray:
        """Retorna apenas a probabilidade de churn (classe 1)"""
        probabilities = self.predict_proba(X)
        return probabilities[:, 1]  # Probabilidade da classe positiva (churn)
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Avalia o modelo no conjunto de teste com métricas abrangentes
        
        Args:
            X_test: Features de teste
            y_test: Target de teste
            
        Returns:
            Dict com métricas de avaliação
        """
        
        logger.info("📊 Avaliando modelo no conjunto de teste...")
        
        # Predições
        y_pred = self.predict(X_test)
        y_pred_proba = self.get_churn_probability(X_test)
        
        # Métricas principais
        auc_score = roc_auc_score(y_test, y_pred_proba)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Métricas de negócio
        true_positives = conf_matrix[1, 1]
        false_positives = conf_matrix[0, 1]
        false_negatives = conf_matrix[1, 0]
        true_negatives = conf_matrix[0, 0]
        
        # Calcular métricas customizadas
        sensitivity = true_positives / (true_positives + false_negatives)  # Recall
        specificity = true_negatives / (true_negatives + false_positives)
        precision = true_positives / (true_positives + false_positives)
        
        metrics = {
            # Métricas básicas
            'auc_score': auc_score,
            'accuracy': class_report['accuracy'],
            'precision': precision,
            'recall': sensitivity,
            'specificity': specificity,
            'f1_score': class_report['1']['f1-score'],
            
            # Matriz de confusão
            'confusion_matrix': conf_matrix.tolist(),
            'true_positives': int(true_positives),
            'false_positives': int(false_positives),
            'false_negatives': int(false_negatives),
            'true_negatives': int(true_negatives),
            
            # Métricas de negócio
            'customers_correctly_identified_as_churn': int(true_positives),
            'customers_wrongly_flagged_as_churn': int(false_positives),
            'churn_customers_missed': int(false_negatives),
            'loyal_customers_correctly_identified': int(true_negatives),
            
            # Relatório completo
            'classification_report': class_report
        }
        
        # Log métricas no MLflow se em uma run ativa
        try:
            mlflow.log_metrics({
                'test_auc': auc_score,
                'test_accuracy': metrics['accuracy'],
                'test_precision': metrics['precision'],
                'test_recall': metrics['recall'],
                'test_f1': metrics['f1_score'],
                'test_specificity': metrics['specificity']
            })
        except:
            pass  # MLflow não está ativo
        
        logger.info("✅ Avaliação concluída!")
        logger.info(f"📊 Métricas principais:")
        logger.info(f"   • AUC: {auc_score:.4f}")
        logger.info(f"   • Acurácia: {metrics['accuracy']:.4f}")
        logger.info(f"   • Precisão: {metrics['precision']:.4f}")
        logger.info(f"   • Recall: {metrics['recall']:.4f}")
        logger.info(f"   • F1-Score: {metrics['f1_score']:.4f}")
        
        return metrics
    
    def plot_feature_importance(self, top_n: int = 20, save_path: str = None) -> None:
        """
        Plota gráfico de importância das features
        
        Args:
            top_n: Número de features mais importantes para mostrar
            save_path: Caminho para salvar o gráfico
        """
        if self.feature_importance is None:
            raise ValueError("❌ Modelo não foi treinado ainda")
        
        plt.figure(figsize=(12, 8))
        top_features = self.feature_importance.head(top_n)
        
        # Criar gráfico horizontal
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
        bars = plt.barh(range(len(top_features)), top_features['importance'], color=colors)
        
        # Configurar gráfico
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importância da Feature')
        plt.title(f'Top {top_n} Features Mais Importantes - Random Forest\nModelo de Predição de Churn')
        plt.gca().invert_yaxis()
        
        # Adicionar valores nas barras
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        
        # Salvar se especificado
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"📊 Gráfico salvo em: {save_path}")
        
        # Log no MLflow se ativo
        try:
            mlflow.log_figure(plt.gcf(), "feature_importance.png")
        except:
            pass
        
        plt.show()
    
    def plot_roc_curve(self, X_test: pd.DataFrame, y_test: pd.Series, save_path: str = None) -> None:
        """
        Plota curva ROC
        
        Args:
            X_test: Features de teste
            y_test: Target de teste  
            save_path: Caminho para salvar o gráfico
        """
        y_pred_proba = self.get_churn_probability(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        plt.figure(figsize=(10, 8))
        
        # Plotar curva ROC
        plt.plot(fpr, tpr, color='darkorange', lw=3, 
                label=f'ROC Curve (AUC = {auc_score:.3f})')
        
        # Linha de referência (classificador aleatório)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier (AUC = 0.500)')
        
        # Configurar gráfico
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Taxa de Falsos Positivos (1 - Especificidade)', fontsize=12)
        plt.ylabel('Taxa de Verdadeiros Positivos (Sensibilidade)', fontsize=12)
        plt.title('Curva ROC - Modelo de Predição de Churn\nRandom Forest Classifier', fontsize=14)
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Adicionar anotações
        plt.text(0.6, 0.2, f'AUC = {auc_score:.3f}', fontsize=16, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        
        plt.tight_layout()
        
        # Salvar se especificado
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"📊 Curva ROC salva em: {save_path}")
        
        # Log no MLflow se ativo
        try:
            mlflow.log_figure(plt.gcf(), "roc_curve.png")
        except:
            pass
        
        plt.show()
    
    def save_model(self, file_path: str) -> None:
        """
        Salva o modelo treinado e metadados
        
        Args:
            file_path: Caminho para salvar o modelo
        """
        if self.model is None:
            raise ValueError("❌ Modelo não foi treinado ainda")
        
        model_data = {
            'model': self.model,
            'feature_importance': self.feature_importance,
            'best_params': self.best_params,
            'training_metrics': self.training_metrics,
            'model_type': 'RandomForestClassifier',
            'trained_at': pd.Timestamp.now().isoformat()
        }
        
        joblib.dump(model_data, file_path)
        logger.info(f"💾 Modelo salvo em: {file_path}")
    
    def load_model(self, file_path: str) -> None:
        """
        Carrega um modelo salvo
        
        Args:
            file_path: Caminho do modelo salvo
        """
        model_data = joblib.load(file_path)
        
        self.model = model_data['model']
        self.feature_importance = model_data.get('feature_importance')
        self.best_params = model_data.get('best_params')
        self.training_metrics = model_data.get('training_metrics', {})
        
        logger.info(f"📂 Modelo carregado de: {file_path}")
        if self.training_metrics:
            logger.info(f"📊 Métricas de treinamento: {self.training_metrics}")

# Exemplo de uso
if __name__ == "__main__":
    # Demonstração do modelo
    logger.info("🚀 Demonstração do Modelo Random Forest para Churn")
    
    # Dados de exemplo (em produção viria do preprocessamento)
    from sklearn.datasets import make_classification
    
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=15, 
        n_redundant=5, n_clusters_per_class=1, 
        weights=[0.7, 0.3], random_state=42
    )
    
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name='churn')
    
    # Dividir em treino e teste
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_series, test_size=0.2, random_state=42, stratify=y_series
    )
    
    # Treinar modelo
    rf_model = ChurnRandomForestModel()
    training_metrics = rf_model.train(X_train, y_train, hyperparameter_tuning=False)
    
    # Avaliar modelo
    test_metrics = rf_model.evaluate(X_test, y_test)
    
    print("\n🎯 Resultados do Modelo:")
    print(f"📊 AUC: {test_metrics['auc_score']:.4f}")
    print(f"📊 Acurácia: {test_metrics['accuracy']:.4f}")
    print(f"📊 Precisão: {test_metrics['precision']:.4f}")
    print(f"📊 Recall: {test_metrics['recall']:.4f}")
    print(f"📊 F1-Score: {test_metrics['f1_score']:.4f}")
    
    print("\n✅ Modelo Random Forest pronto para produção!")