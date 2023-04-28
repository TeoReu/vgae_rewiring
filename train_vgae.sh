for model in "GCN"
do
  for split_graph in "_"
  do
    for layers in 1 2 4 8
    do
      for alpha in  -10 - 5 0.1 0.5 1 2
      do
        for transform in True False
        do
          python zinc_vgae.py --model $model --split_graph $split_graph  --layers $layers --file_name "results_vgae" --alpha $alpha --transform $transform
        done
      done
    done
  done
done
