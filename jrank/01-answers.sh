# python gen_model_answer.py --bench_name rakuda_v2 --model-path /models/shisa-7b-v1 --model-id shisa-7b-v1 --conv_template ./templates/shisa.json
# time python gen_model_answer.py --bench_name rakuda_v2 --model-path /models/shisa-7b-v1 --model-id shisa-7b-v1 --conv_template ./templates/shisa.json
# time python gen_model_answer.py --bench_name rakuda_v2 --model-path /models/shisa-7b-v1 --model-id shisa-7b-v1 --conv_template ./templates/shisa.json
# time python gen_model_answer.py --bench_name rakuda_v2 --model-path /models/llm/hf/moneyforward_houou-instruction-7b-v1 --model-id houou-instruction-7b-v1 --conv_template ./templates/houou.json
time python gen_model_answer.py --bench_name rakuda_v2 --model-path /models/shisa-7b-v1 --model-id shisa-7b-v1 --conv_template ./templates/shisa.json --repetition_penalty 1.05
time python gen_model_answer.py --bench_name rakuda_v2 --model-path NTQAI/chatntq-ja-7b-v1.0 --model-id chatntq-ja-7b-v1.0 --conv_template ./templates/chatntq2.json --repetition_penalty 1.05
