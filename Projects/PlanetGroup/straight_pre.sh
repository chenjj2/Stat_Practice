for i in 25. 50. 100. 200.
do
	for j in 1 2 3 4 5
	do
	python straight_pre.py --temperature $i --lindir lin$j &
	done
	python straight_ic.py --temperature $i --lindir lin0

	wait
done


