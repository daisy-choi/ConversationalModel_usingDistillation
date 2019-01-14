for i in $(seq 0 3)
do
  python translate.py -model distill_files/model$i.pt -src temp -output temp_files/test_output_$i.txt -replace_unk -verbose -gpu 0
done
