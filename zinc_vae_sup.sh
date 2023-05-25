for threshold in 0.0 0.4
do
    for alpha in  -10.0 -5.0 0.0 1.0 5.0
    do
      python peptides_vae_sup.py --alpha $alpha --threshold $threshold --pe 20 -- conv "GINE"
  done
done