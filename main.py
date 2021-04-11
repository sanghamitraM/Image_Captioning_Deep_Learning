from data_helper import *
from Dataset import *
import matplotlib.pyplot as plt
from base_mode import *
from build_vocab import Vocabulary
#from gensim.models import Word2Vec
from torch.nn.utils.rnn import pack_padded_sequence
import pickle
import logging

logging.basicConfig(level=logging.INFO,filename='baseline_model.log',filemode='w')
def main(data_path,batch_size,embed_size,learning_rate,num_epochs,hidden_size,num_layers):
    #data_path = 'd:\\IG Data\\'
    root = os.getcwd()
    file_dict = create_file_paths_dict(data_path)
#print(file_dict)
    data_df = create_data_csv(file_dict)
    os.chdir(root)

    transformer = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    with open('vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)

    dataset = IG_img_caption_dataset(
        csv_file = data_df,
        type_prof = 'trial',
        vocab=vocab,
        transform = transformer)

    dataloader = DataLoader(dataset=dataset,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=collate_fn,
        shuffle=True)

#https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning
# Build the models
#torch.cuda.clear_memory_allocated()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#torch.cuda.reset_max_memory_allocated(device)
#print(torch.cuda.max_memory_reserved(device))

    encoder = EncoderCNN(embed_size).to(device)
    decoder = DecoderRNN(embed_size, hidden_size, len(vocab), num_layers).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()

    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    
    # Train the models
    total_step = len(dataloader)
    for epoch in range(num_epochs):
        for i, (images, captions, lengths) in enumerate(dataloader):
            images = images.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            
            # Forward, backward and optimize
            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            loss = criterion(outputs, targets)
            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            optimizer.step()

            # Print log info
            logs = 'Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch, num_epochs, i, total_step, loss.item(), np.exp(loss.item())))
            print(logs)
            logging.info(logs)
        
            torch.save(decoder.state_dict(), 'Saved Models/baseline/decoder-{}-{}.ckpt'.format(epoch+1, i+1))
            torch.save(encoder.state_dict(), 'Saved Models/baseline/encoder-{}-{}.ckpt'.format(epoch+1, i+1))

if __name__=='__main__':
    
    main(data_path = '/googledrive/Shared drives/IDS 576/Project',
         batch_size=5,
         embed_size=100,
         learning_rate=0.1,
         num_epochs=10,
         hidden_size=50,
         num_layers=5)