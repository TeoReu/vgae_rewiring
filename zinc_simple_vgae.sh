for model in "GCN" "GIN" "PNA"
do
  for transform in "no" "random_walk" "laplacian"
  do
    for layers in 1 2 4 6 8
    do
      python zinc_simple_vgae.py --model $model --layers $layers --file_name "zinc_simple_vgae" --transform $transform
    done
  done
done