import torch

# Các đường dẫn dữ liệu và tên tệp
DATA_DIR = 'data'
SP_DIR = f'{DATA_DIR}/sp'  # Thư mục chứa model và vocal của SentencePiece
SRC_DIR = 'src'   # Thư mục chứa dữ liệu nguồn       
TRG_DIR = 'trg'   # Thư mục chứa dữ liệu đích

# Tên các tệp dữ liệu
SRC_RAW_DATA_NAME = 'raw_data.src'
TRG_RAW_DATA_NAME = 'raw_data.trg'
TRAIN_NAME = 'train.txt'
VALID_NAME = 'valid.txt'
TEST_NAME = 'test.txt'

# Các token đặc biệt
pad_id = 0    # Đệm các câu ngắn cho bằng độ dài batch
sos_id = 1    # Tín hiệu bắt đầu để Decoder sinh từ
eos_id = 2    # Tín hiệu báo hiệu kết thúc câu
unk_id = 3    # Từ không có trong từ điển

# Cấu hình SentencePiece
src_model_prefix = 'src_sp'
trg_model_prefix = 'trg_sp'
sp_vocab_size = 16000   # Kích thước từ điển
character_coverage = 1.0   # Độ phủ ký tự
model_type = 'unigram'   # Loại mô hình SentencePiece 'unigram' hoặc 'bpe' hoặc 'word' hoặc 'char'

# Các siêu tham số
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  # Phát hiện và sử dụng GPU nếu có
learning_rate = 1e-4   # Tốc độ học
batch_size = 64   # Kích thước batch
seq_len = 128   # Độ dài chuỗi tối đa
num_heads = 8   # Số lượng Attention Heads trong Multi-Head Attention
num_layers = 6   # Số lượng lớp chồng lên nhau trong Encoder và Decoder stack
d_model = 512   # Kích thước embedding và hidden state
d_ff = 2048   # Kích thước của Feed Forward layer
d_k = d_model // num_heads  # Kích thước của mỗi Attention Head
drop_out_rate = 0.1    # Tỷ lệ Dropout
num_epochs = 5   # Số epoch huấn luyện
beam_size = 5   # Kích thước beam search
ckpt_dir = 'saved_model'   # Thư mục lưu trữ checkpoint mô hình

# Cấu hình Attention
attention_type = 'luong' # 'bahdanau' or 'luong' or 'scaled_dot_product'
start_epoch = 1  # Epoch bắt đầu (dùng khi resume training từ checkpoint)