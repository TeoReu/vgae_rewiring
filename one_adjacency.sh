for model in "simple"
do
  for layers in 1 2 4
  do
    for transform in 0 1
    do
      for nr in 234 123
      do
        python one_adjacency_zinc.py --model $model --layers $layers --file_name "one_adj_with_edge_attr" --transform $transform --nr $nr
      done
    done
  done
done