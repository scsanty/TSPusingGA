To run the code:<br>
python main.py <inst_file> <nRuns> <nIterations> <popSize> <initalisation> <Pc> <Pm> <truncationPercentage> <elitismPercentatge> <br>
e.g.: - <br>
python main.py TSP_dataset/inst-19.tsp 10 500 100 1 0.8 0.05 0.25 0.0<br><br>

Rules sanity checks are performed in the code:
<pre>
<inst_file>			May use absolute/relative path & it should not have spaces in between
<nRuns>				Should not be fractions or less than 0
<nIterations>			Should not be fractions or less than 0
<popSize>			Should not be fractions or less than 1

<Pc> <Pm>			Should be a fraction betweeen 0 or 1 as these are probabilities
<truncationPercentage>		even if the name suggests percentage
<elitismPercentatge>

Libraries:
Matplotlib			external library, installation required for visualisations,
				wouldn't fail if not found
math, sys, random		internal libraries, no installation required.
</pre>
