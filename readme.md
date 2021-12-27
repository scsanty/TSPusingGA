To run the code:<br>
python main.py &lt;inst_file&gt; &lt;nRuns&gt; &lt;nIterations&gt; &lt;popSize&gt; &lt;initalisation&gt; &lt;Pc&gt; &lt;Pm&gt; &lt;truncationPercentage&gt; &lt;elitismPercentatge&gt; <br>
e.g.: - <br>
python main.py TSP_dataset/inst-19.tsp 10 500 100 1 0.8 0.05 0.25 0.0<br><br>

Rules sanity checks are performed in the code:
<pre>
&lt;inst_file&gt;			May use absolute/relative path & it should not have spaces in between
&lt;nRuns&gt;				Should not be fractions or less than 0
&lt;nIterations&gt;			Should not be fractions or less than 0
&lt;popSize&gt;			Should not be fractions or less than 1

&lt;Pc&gt; &lt;Pm&gt;			Should be a fraction betweeen 0 or 1 as these are probabilities
&lt;truncationPercentage&gt;		even if the name suggests percentage
&lt;elitismPercentatge&gt;

Libraries:
Matplotlib			external library, installation required for visualisations,
				wouldn't fail if not found
math, sys, random		internal libraries, no installation required.
</pre>
