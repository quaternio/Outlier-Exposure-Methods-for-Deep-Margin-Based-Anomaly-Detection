import torch.nn as nn
import torch.backends.cudnn as cudnn
from tqdm import tqdm

torch.manual_seed(0)
cudnn.deterministic = True
cudnn.benchmark = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'

alphas = [0.01, 0.03, 0.1, 0.3]
learning_rates = [3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
learning_rate = 1e-3
alpha = 0.01
lambda_l2 = 1e-5
batch_size = 8

num_known_classes = 4

train_nom_dataset  = torch.load('s_z_train_nom_dataset.pt')
train_anom_dataset = torch.load('s_z_train_anom_dataset.pt')
test_anom_dataset  = torch.load('s_z_test_anom_dataset.pt')
test_nom_dataset   = torch.load('s_z_test_nom_dataset.pt')

train_anom_dataloader = \
        torch.utils.data.DataLoader(
            train_anom_dataset,
            batch_size = batch_size,
            shuffle = True,
            num_workers = 0
        )
test_nom_dataloader   = \
        torch.utils.data.DataLoader(
            test_nom_dataset,
            batch_size = batch_size,
            shuffle = True,
            num_workers = 0
        )

novel_loss = nn.CrossEntropyLoss()
#known_loss = nn.CrossEntropyLoss(label_smoothing=alpha)

optimizer = torch.optim.SGD(subject_classifier.parameters(), lr=learning_rate, weight_decay=lambda_l2)

m = torch.nn.Softmax(dim=1)
one_hot = torch.nn.functional.one_hot

subject_classifier.train()
subject_classifier.to(device)

# Since there are only 73 examples in train_anom, this should only utilize
# 73 of the known in instances in the test set, leaving the rest for testing.
# There are 721 nominal examples in the test nom set total.

losses = []
lr_losses = []
alpha_losses = []
num_batches = len(train_anom_dataset) // batch_size

for alpha in [0.01]:
    #optimizer = torch.optim.SGD(subject_classifier.parameters(), lr=lr, weight_decay=lambda_l2)
    known_loss = nn.CrossEntropyLoss(label_smoothing=alpha)
    losses = []
    for i in range(1):
        for j, ((anom_X, anom_y), (nom_X, nom_y)) in tqdm(enumerate(zip(train_anom_dataloader, test_nom_dataloader))):

            anom_X = anom_X.to(device)
            nom_X  = nom_X.to(device)

            # Keep this line uncommented for detection feedback
            nom_y = torch.zeros_like(nom_y)

            # Feed forward
            anom_predictions = m(subject_classifier(anom_X)[:,1:5])
            nom_predictions  = m(subject_classifier(nom_X)[:,1:5])
            
            # Keep this line uncommented for detection feedback
            nom_y_orig = nom_y
            nom_y = torch.argmax(nom_predictions, dim=1)+1 #[:,1:5]

            # If the instance is anomalous, use uniform labeling
            # NOTE: label matrix is (batch_size x num_nominal_classes)
            anom_y = (1/num_known_classes)*torch.ones((len(anom_y),num_known_classes))

            # Converting the nominal labels to be one-hot. Note that these
            # will be label-smoothed when loss is computed
            nom_y = torch.squeeze(one_hot(nom_y-1, num_known_classes).float()) #nom_y = torch.argmax(nom_predictions, dim=1)+1 #[:,1:5]

            anom_y = anom_y.to(device)
            nom_y = nom_y.to(device)

            # Compute loss. Note that arguments are (input, target).
            #
            # Input needs to be (batch_size x num_classes).
            # Target needs to be same shape as input since it contains class
            # probabilities.
            loss = novel_loss(anom_predictions, anom_y) + known_loss(nom_predictions, nom_y)
            losses.append(loss)

            # Zero the optimizer's gradients
            optimizer.zero_grad()

            # Backpropagate to compute the gradients
            # of the loss with respect to our learnable
            # parameters
            loss.backward()

            # Update the learnable parameters
            optimizer.step()
    
    alpha_losses.append(losses)
