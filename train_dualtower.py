import os, sys
import numpy as np
import tensorflow as tf
import drug_encoder
import protein_encoder
import dualtower

from sklearn.model_selection import KFold

def pearson_r(y_true, y_pred):
    a = y_true - np.mean(y_true)
    b = y_pred - np.mean(y_pred)
    up = np.sum(a * b)
    down = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return up / down


seed = 123
tf.random.set_seed(seed)
fold_id = int(sys.argv[3])

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True) 
    except RuntimeError as e:
        print(e)


epochs = 300
d_model = 128
n_2d_layers = 2
n_3d_layers = 1

dataset = os.path.join("data", "davis", 'davis_kinase')


smiles_vocab_size = 40
amino_vocab_size = 30
max_smiles_len = 80
max_amino_len = 800

p_encoder = protein_encoder.ProteinConvEncoder(128, amino_vocab_size, max_amino_len, kernel_size=7)


drug_feat = 'GCN3D'
d_encoder = drug_encoder.Drug3DEncoder(d_model, n_layers_2d=n_2d_layers, n_layers_3d=n_3d_layers)
# drug_a, drug_s, drug_x, protein_seq, label
drug_data = [   
                np.load(f"{dataset}.csv_a.npy", allow_pickle=True),
                np.load(f"{dataset}.csv_s.npy", allow_pickle=True),
                np.load(f"{dataset}.csv_x.npy", allow_pickle=True)
                ]
prot_data = np.load(f"{dataset}.csv_prot.npy", allow_pickle=True)
label = np.load(f"{dataset}.csv_y.npy", allow_pickle=True)

# check data
_ = d_encoder([_temp[:10] for _temp in drug_data])
print(f"Encoded drug dimension: {_.shape[1:]} <Length, Dimension> ")
_ = p_encoder(prot_data[:10], training=False)
print(f"Encoded protein dimension: {_.shape[1:]} <Length, Dimension> ")

del _
print("Data loaded")   


model = dualtower.DualTower(
            d_encoder, p_encoder,
            fc_dims = [512, 128, 32], dropout_rate=0.1
            )
#drug_data, protein_data
_ = model([_[:10] for _ in drug_data], prot_data[:10], training=False)
print("Model initialized")
del _
learning_rate = 1e-4
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

ckpt_path = f"{drug_feat}_{seed}_{fold_id}_{d_model}_{n_2d_layers}_{n_3d_layers}"
checkpoint_path = os.path.join('.', 'checkpoints_dualtower_para', ckpt_path)
os.makedirs(checkpoint_path, exist_ok=True)
ckpt = tf.train.Checkpoint(model=model,
                           optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=10)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    print(ckpt_manager.latest_checkpoint)
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print ('Latest checkpoint restored!!', ckpt_manager.latest_checkpoint)

mse_loss = tf.keras.losses.MeanSquaredError()
loss_tracker = tf.keras.metrics.Mean(name='loss')

@tf.function
def train_step(_drug_data, _protein_data, y_true):
    with tf.GradientTape() as tape:
        y_pred = model(_drug_data, _protein_data, training=True)
        loss = mse_loss(y_true, y_pred)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    loss_tracker(loss)
    return y_true, y_pred

@tf.function
def test_step(_drug_data, _protein_data, y_true):
    with tf.GradientTape() as tape:
        y_pred = model(_drug_data, _protein_data, training=False)
        loss = mse_loss(y_true, y_pred)
    
    loss_tracker(loss)
    return y_true, y_pred


dataset_size = len(label)

kfold = KFold(n_splits=5, shuffle=True, random_state=seed)

train_idx, test_idx = [(_i, _j) for (_i, _j) in kfold.split(np.arange(dataset_size))][fold_id]

train_size = len(train_idx)
test_size = len(test_idx)

np.random.seed(seed)
batch_size = 64
train_batch_num = int(np.ceil(train_size / batch_size))
test_batch_num = int(np.ceil(test_size / batch_size))


flog = open(f"dualtower_davis_{drug_feat}_{seed}_{fold_id}_{d_model}_{n_2d_layers}_{n_3d_layers}.csv", 'w')
print("epoch,train_pcc,train_loss,test_pcc,test_loss", file=flog)
flog.close()
for epoch in range(epochs):
    flog = open(f"dualtower_davis_{drug_feat}_{seed}_{fold_id}_{d_model}_{n_2d_layers}_{n_3d_layers}.csv", 'a')
    print("==================")
    print("Epoch ", epoch + 1)
    print(f"{epoch + 1}", end=',', file=flog)
    train_idx = np.random.permutation(train_idx)
    
    loss_tracker.reset_states()
    y_t_list = []
    y_p_list = []
    for train_batch_i in range(train_batch_num):
        print(f"Batch: {train_batch_i}", end='\r')
        current_train_idx = train_idx[train_batch_i * batch_size: (train_batch_i + 1) * batch_size]
        
        y_t, y_p = train_step(
            [_[current_train_idx] for _ in drug_data], 
            prot_data[current_train_idx],
            label[current_train_idx])
        
        y_t_list += y_t.numpy().squeeze().tolist()
        y_p_list += y_p.numpy().squeeze().tolist()
        
        
    print("\nTraining")
    y_t_list = np.array(y_t_list)
    y_p_list = np.array(y_p_list)
    pcc = pearson_r(y_t_list, y_p_list)
    print(f"PCC: {pcc:0.4f} Loss: {loss_tracker.result().numpy():.4f}")
    print(f"{pcc:0.4f},{loss_tracker.result().numpy():.4f}", end=',', file=flog)
    # print(y_t, y_p.numpy().dtype)
    ckpt_save_path = ckpt_manager.save()
    
    loss_tracker.reset_states()
    y_t_list = []
    y_p_list = []
    for test_batch_i in range(test_batch_num):
        print(f"Batch: {test_batch_i}", end='\r')
        current_test_idx = test_idx[test_batch_i * batch_size: (test_batch_i + 1) * batch_size]
        
        y_t, y_p = test_step(
            [_[current_test_idx] for _ in drug_data], 
            prot_data[current_test_idx],
            label[current_test_idx])
        
        y_t_list += y_t.numpy().squeeze().tolist()
        y_p_list += y_p.numpy().squeeze().tolist()
        
    print("\nTest")
    y_t_list = np.array(y_t_list)
    y_p_list = np.array(y_p_list)
    pcc = pearson_r(y_t_list, y_p_list)
    print(f"PCC: {pcc:0.4f} Loss: {loss_tracker.result().numpy():.4f}")
    print(f"{pcc:0.4f},{loss_tracker.result().numpy():.4f}", file=flog)
    flog.close()


