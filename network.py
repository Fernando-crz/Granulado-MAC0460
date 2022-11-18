import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

from matplotlib import pyplot as plt

class NN():
	def __init__(self, operator_size):
		super(NN, self).__init__()
		self.operation_sequence = nn.Sequential(
			nn.Linear(operator_size, operator_size*2),
			nn.ReLU(),
			nn.Linear(operator_size*2, 1)
		)

	def forward(self, x):
		result = self.operation_sequence(x)
		return result

def check_accuracy(loader, model):
	# to-do
	num_correct = 0
	num_samples = 0
	model.eval()

	# desligando variacao de gradientes para avaliacao do modelo:
	with torch.no_grad():
		for data, target in loader:
			scores = model(data)

def main():
	# definindo o dispositivo utilizado
	device = "cuda" if torch.cuda.is_available() else "cpu"
	print(f"Usando o dispositivo {device}")

	# carregando o dataset a ser utilizado.

	# definindo variaveis importantes
	operator_size = 5
	learning_rate = 0.01
	batch_size = 16
	num_epochs = 2

	# criando modelo de treinamento
	model = NN(operator_size).to(device)

	# iniciando funcoes de perda e otimizador
	loss = nn.CrossEntropyLoss()  # testar tambem com MSE
	optimizer = optim.RAdam(model.parameters(), lr=learning_rate)

	# treinar modelo
	for epoch in range(num_epochs):
		for i, (data, target) in enumerate(train_data):
			data.to(device)
			target.to(device)

			preds = model(data)
			error = loss(preds, target)

			optimizer.zero_grad()
			error.backward()

			optimizer.step()

if __name__ == "__main__":
	main() 