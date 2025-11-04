TEST_PATH=/home/sriram.sk/desktop/rpad/_data_sriram_hoi4d_hoi4d_data_ZY20210800003_H3_C14_N42_S207_s05_T2_vid/

python hand2sim.py -ip $TEST_PATH --env_name robohive --emb pjaw
python hand2sim.py -ip $TEST_PATH --env_name isaacsim --emb pjaw
python hand2sim.py -ip $TEST_PATH --env_name isaacsim --emb allegro
