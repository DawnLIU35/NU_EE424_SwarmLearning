Project of EE424 distributed optimization.
Use LISA Traffic Light dataset. https://www.kaggle.com/mbornoe/lisa-traffic-light-dataset

docker login hub.myenterpriselicense.hpe.com -u liu1103159436@gmail.com -p hpe_eval
export DOCKER_CONTENT_TRUST=1
cd /mnt/d/Northwestern_Local/22Winter/EE424_DistributedOptimization/NU_EE424_SwarmLearning
./swarm-learning/bin/run-apls --apls-port=5814

Run on single SN node and 2 SL nodel:
(open a new termainal to run each of the following command)

./LISA_Traffic_Light4/bin/init-workspace -e LISA_Traffic_Light4 -i 172.22.16.143 -d /mnt/d/Northwestern_Local/22Winter/EE424_DistributedOptimization/NU_EE424_SwarmLearning/examples

../swarm-learning/bin/run-sl --name node1-sl --network LISA_Traffic_Light4-net --host-ip node1-sl --sn-ip node-sn -e MAX_EPOCHS=5 --apls-ip 172.22.16.143 --serverAddress node-spire -genJoinToken --data-dir /mnt/d/Northwestern_Local/22Winter/EE424_DistributedOptimization/NU_EE424_SwarmLearning/examples/ws-LISA_Traffic_Light4/node1/app-data --model-dir /mnt/d/Northwestern_Local/22Winter/EE424_DistributedOptimization/NU_EE424_SwarmLearning/examples/ws-LISA_Traffic_Light4/node1/model --model-program LISA_pyt.py --sl-platform PYT --gpu 0

../swarm-learning/bin/run-sl --name node2-sl --network LISA_Traffic_Light4-net --host-ip node2-sl --sn-ip node-sn -e MAX_EPOCHS=5 --apls-ip 172.22.16.143 --serverAddress node-spire -genJoinToken --data-dir /mnt/d/Northwestern_Local/22Winter/EE424_DistributedOptimization/NU_EE424_SwarmLearning/examples/ws-LISA_Traffic_Light4/node2/app-data --model-dir /mnt/d/Northwestern_Local/22Winter/EE424_DistributedOptimization/NU_EE424_SwarmLearning/examples/ws-LISA_Traffic_Light4/node2/model --model-program LISA_pyt.py --sl-platform PYT --gpu 0

sudo ./LISA_Traffic_Light4/bin/del-workspace -e LISA_Traffic_Light4 -d /mnt/d/Northwestern_Local/22Winter/EE424_DistributedOptimization/NU_EE424_SwarmLearning/examples