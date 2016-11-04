datapath=$1 #path to data
folds=$2 #how many folds in the data
#rescat=$3 #path to result category

corerescat=$(echo $datapath | tr -dc '[:alnum:]\n\r')
dt=$(date +%Y%m%d%H%M%S)
rescat=$dt""$corerescat
rescat="/data/acp13ml/seqhawkes_results/"$rescat
echo $rescat
mkdir $rescat

RECEPIENT_EMAIL="m.lukasik@sheffield.ac.uk" #you might want to change this e-mail address
RESULT_FILE="res_tmp"$corerescat".txt"

python qsub_runs.py $folds $datapath 0 easy $rescat
echo "MICRO:\n" > $RESULT_FILE
python gather_results.py $rescat micro >> $RESULT_FILE
echo "MACRO:\n" >> $RESULT_FILE
python gather_results.py $rescat macro >> $RESULT_FILE

cat $RESULT_FILE | mutt -a $RESULT_FILE -s "Sequence HP results from "$rescat -- $RECEPIENT_EMAIL
