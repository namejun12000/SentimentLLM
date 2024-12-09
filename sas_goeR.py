import os
import sys

# 환경 변수 설정: Tokenizer 병렬 처리 비활성화
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 작업 디렉터리를 프로젝트의 루트로 설정
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 이후 SASRec 경로를 설정하고 import
sas_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'SASRec'))
if sas_dir not in sys.path:
    sys.path.append(sas_dir)

# 이후 goeRobert 경로를 설정하고 import
goe_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'goeRobert'))
if goe_dir not in sys.path:
    sys.path.append(goe_dir)

from SASRec.sasrec_train_infer import get_args, train_and_infer
from goeRobert.goeRoberta import run_sentiment_analysis

def main():
    
    # Goemotions + RoBertA
    sentiment_input = 'data/only_review.txt'  # Input file path
    sentiment_output = 'goeRobert/sentiment_strength_results.txt'  # Output file path
    
    # 감성 분석 실행
    run_sentiment_analysis(sentiment_input, sentiment_output)

    # SASRec
    args = get_args()  # 명령줄 인자를 받아옴

    if not os.path.isdir(args.dataset + '_' + args.train_dir):
        os.makedirs(args.dataset + '_' + args.train_dir)
    with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
        f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
    f.close()
    train_and_infer(args)  # 훈련 및 추론 실행

if __name__ == "__main__":
    main()