
import os, sys
import numpy as np
import tensorflow as tf
import drug_encoder
import protein_encoder
import transformer


seed = 123
tf.random.set_seed(seed)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True) 
    except RuntimeError as e:
        print(e)


epochs = 50
d_model = 128
n_2d_layers=2
n_3d_layers=1
dataset = os.path.join("data", "gpcr", 'GPCR')
smiles_vocab_size = 40
max_smiles_len = 80

p_encoder = protein_encoder.ConvGLUEncoder(d_model, kernel_size=7)
    
drug_feat = 'GCN3D'
d_encoder = drug_encoder.Drug3DEncoder(d_model, n_layers_2d=n_2d_layers, n_layers_3d=n_3d_layers)
# drug_a, drug_s, drug_x, drug_mask, protein_embed, protein_mask, label
train_drug_data = [
                      np.load(f"{dataset}_train.txt_a.npy", allow_pickle=True),
                      np.load(f"{dataset}_train.txt_s.npy", allow_pickle=True),
                      np.load(f"{dataset}_train.txt_x.npy", allow_pickle=True)
                      ]
train_drug_mask = np.load(f"{dataset}_train.txt_drug_mask.npy", allow_pickle=True)
train_prot_data = np.load(f"{dataset}_train.txt_prot.npy", allow_pickle=True)
train_prot_mask =  np.load(f"{dataset}_train.txt_prot_mask.npy", allow_pickle=True)
train_label = np.load(f"{dataset}_train.txt_y.npy", allow_pickle=True)

test_drug_data = [
                      np.load(f"{dataset}_test.txt_a.npy", allow_pickle=True),
                      np.load(f"{dataset}_test.txt_s.npy", allow_pickle=True),
                      np.load(f"{dataset}_test.txt_x.npy", allow_pickle=True)
                      ]
test_drug_mask = np.load(f"{dataset}_test.txt_drug_mask.npy", allow_pickle=True)
test_prot_data = np.load(f"{dataset}_test.txt_prot.npy", allow_pickle=True)
test_prot_mask =  np.load(f"{dataset}_test.txt_prot_mask.npy", allow_pickle=True)
test_label = np.load(f"{dataset}_test.txt_y.npy", allow_pickle=True)

# check data
_ = d_encoder([_temp[:10] for _temp in train_drug_data])
_ = d_encoder([_temp[:10] for _temp in test_drug_data])
print(f"Encoded drug dimension: {_.shape[1:]} <Length, Dimension> ")

_ = p_encoder(train_prot_data[:10], training=False)
_ = p_encoder(test_prot_data[:5], training=False)
print(f"Encoded protein dimension: {_.shape[1:]} <Length, Dimension> ")

del _
print("Data loaded")



model = transformer.DrugTransformer(
            d_encoder, p_encoder,
            num_layers=3, d_model=d_model, num_heads=8, dff=512,
            fc_units=256
            )
#drug_data, protein_data, drug_mask, protein_mask
_ = model([_[:10] for _ in test_drug_data], test_prot_data[:10], test_drug_mask[:10], test_prot_mask[:10], training=False)
print("Model initialized")


del _
learning_rate = 1e-4
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

ckpt_path = drug_feat + '_' + f"{seed}_{d_model}_{n_2d_layers}_{n_3d_layers}"
checkpoint_path = os.path.join('.', 'checkpoints_transformer_para', ckpt_path)
os.makedirs(checkpoint_path, exist_ok=True)
ckpt = tf.train.Checkpoint(model=model,
                           optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=10)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    print(ckpt_manager.latest_checkpoint)
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print ('Latest checkpoint restored!!', ckpt_manager.latest_checkpoint)

bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)


auroc = tf.keras.metrics.AUC(curve='ROC', name='auroc')
aupr = tf.keras.metrics.AUC(curve='PR', name='aupr')
loss_tracker = tf.keras.metrics.Mean(name='loss')

@tf.function
def train_step(drug_data, protein_data, drug_mask, protein_mask, y_true):
    with tf.GradientTape() as tape:
        y_pred = model(drug_data, protein_data, drug_mask, protein_mask, training=True)
        loss = bce_loss(y_true, y_pred)
        y_pred_sigmoid = tf.math.sigmoid(y_pred)
        
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    auroc(y_true, y_pred_sigmoid)
    aupr(y_true, y_pred_sigmoid)
    loss_tracker(loss)
    return y_true, y_pred_sigmoid

@tf.function
def test_step(drug_data, protein_data, drug_mask, protein_mask, y_true):
    with tf.GradientTape() as tape:
        y_pred = model(drug_data, protein_data, drug_mask, protein_mask, training=False)
        loss = bce_loss(y_true, y_pred)
        y_pred_sigmoid = tf.math.sigmoid(y_pred)
    
    auroc(y_true, y_pred_sigmoid)
    aupr(y_true, y_pred_sigmoid)
    loss_tracker(loss)
    return y_true, y_pred_sigmoid


train_size = len(train_label)
test_size = len(test_label)
np.random.seed(seed)
batch_size = 64
train_batch_num = int(np.ceil(train_size / batch_size))
test_batch_num = int(np.ceil(test_size / batch_size))


flog = open(f"transformer_GPCR_{drug_feat}_{seed}_{d_model}_{n_2d_layers}_{n_3d_layers}.csv", 'w')
print("epoch,train_auc,train_aupr,train_loss,test_auc,test_aupr,test_loss", file=flog)
flog.close()
for epoch in range(epochs):
    flog = open(f"transformer_GPCR_{drug_feat}_{seed}_{d_model}_{n_2d_layers}_{n_3d_layers}.csv", 'a')
    print("==================")
    print("Epoch ", epoch + 1)
    print(f"{epoch + 1}", end=',', file=flog)
    train_idx = np.random.permutation(train_size)
    test_idx = np.random.permutation(test_size)
    
    auroc.reset_states()
    aupr.reset_states()
    loss_tracker.reset_states()
    
    for train_batch_i in range(train_batch_num):
        print(f"Batch: {train_batch_i}", end='\r')
        current_train_idx = train_idx[train_batch_i * batch_size: (train_batch_i + 1) * batch_size]
        
        y_t, y_p = train_step(
            [_[current_train_idx] for _ in train_drug_data], 
            train_prot_data[current_train_idx],
            train_drug_mask[current_train_idx], 
            train_prot_mask[current_train_idx],
            train_label[current_train_idx])
        
    print("\nTraining")
    print(f"AUROC: {auroc.result().numpy():.4f} AUPR: {aupr.result().numpy():.4f} Loss: {loss_tracker.result().numpy():.4f}")
    print(f"{auroc.result().numpy():.4f},{aupr.result().numpy():.4f},{loss_tracker.result().numpy():.4f}", end=',', file=flog)
    # print(y_t, y_p.numpy().dtype)
    ckpt_save_path = ckpt_manager.save()
    
    auroc.reset_states()
    aupr.reset_states()
    loss_tracker.reset_states()
    for test_batch_i in range(test_batch_num):
        print(f"Batch: {test_batch_i}", end='\r')
        current_test_idx = test_idx[test_batch_i * batch_size: (test_batch_i + 1) * batch_size]
        
        y_t, y_p = test_step(
            [_[current_test_idx] for _ in test_drug_data], 
            test_prot_data[current_test_idx],
            test_drug_mask[current_test_idx], 
            test_prot_mask[current_test_idx],
            test_label[current_test_idx])
        
    print("\nTest")
    print(f"AUROC: {auroc.result().numpy():.4f} AUPR: {aupr.result().numpy():.4f} Loss: {loss_tracker.result().numpy():.4f}")
    print(f"{auroc.result().numpy():.4f},{aupr.result().numpy():.4f},{loss_tracker.result().numpy():.4f}", file=flog)
    flog.close()



