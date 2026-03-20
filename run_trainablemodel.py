from financial_forecast.models.trainable_model import TrainableFinancialModel
from financial_forecast.data.aapl.financial_statements import get_financial_statements

model = TrainableFinancialModel()
data = get_financial_statements()
model.train_simple_policies(data, epochs=5000)
model.train_structural_parameters(data, epochs=5000)
