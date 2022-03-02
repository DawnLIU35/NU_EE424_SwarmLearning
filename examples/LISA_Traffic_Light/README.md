Project of EE424 distributed optimization.
Use LISA Traffic Light dataset. https://www.kaggle.com/mbornoe/lisa-traffic-light-dataset

Run on single SN node and 2 SL nodel:

./LISA_Traffic_Light/bin/init-workspace -e LISA_Traffic_Light -i 172.18.75.171 -d /mnt/d/Northwestern_Local/22Winter/EE424_DistributedOptimization/NU_EE424_SwarmLearning/examples

../swarm-learning/bin/run-sl --name node1-sl --network LISA_Traffic_Light-net --host-ip node1-sl --sn-ip node-sn -e MAX_EPOCHS=5 --apls-ip 172.18.75.171 --serverAddress node-spire -genJoinToken --data-dir /mnt/d/Northwestern_Local/22Winter/EE424_DistributedOptimization/NU_EE424_SwarmLearning/examples/ws-LISA_Traffic_Light/node1/app-data --model-dir /mnt/d/Northwestern_Local/22Winter/EE424_DistributedOptimization/NU_EE424_SwarmLearning/examples/ws-LISA_Traffic_Light/node1/model --model-program LISA_pyt.py --sl-platform PYT --gpu 0

../swarm-learning/bin/run-sl --name node2-sl --network LISA_Traffic_Light-net --host-ip node2-sl --sn-ip node-sn -e MAX_EPOCHS=5 --apls-ip 172.18.75.171 --serverAddress node-spire -genJoinToken --data-dir /mnt/d/Northwestern_Local/22Winter/EE424_DistributedOptimization/NU_EE424_SwarmLearning/examples/ws-LISA_Traffic_Light/node2/app-data --model-dir /mnt/d/Northwestern_Local/22Winter/EE424_DistributedOptimization/NU_EE424_SwarmLearning/examples/ws-LISA_Traffic_Light/node2/model --model-program LISA_pyt.py --sl-platform PYT --gpu 0

sudo ./LISA_Traffic_Light/bin/del-workspace -e LISA_Traffic_Light -d /mnt/d/Northwestern_Local/22Winter/EE424_DistributedOptimization/NU_EE424_SwarmLearning/examples