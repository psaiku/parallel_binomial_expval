import MPI
using Distributions

function simBer(t) return(rand(Bernoulli(t[1]))) end

function cal_optvalue(N, S, sigma, K, r, T, opt_type, n_sam, nrep)
	MPI.Init()
	comm = MPI.COMM_WORLD
	id = MPI.Comm_rank(comm)
	np = MPI.Comm_size(comm)
	# since we assume that the asset returns are log-normal with non-constant variance, u, d, and p are time-varying
	dt = T/N
	beta = 1/2*(exp(-r*dt)+exp((r+sigma^2)*dt))
	u = beta + sqrt(beta^2-1)
	d = 1/u
	p = (exp(r*dt)-d)/(u-d)
	probs = fill(p, N)
	u = fill(u, N)
	d = fill(d, N)
	M = log2(np)
	vals = zeros(nrep)
	probs_s = fill(p, convert(Int64, M))
	probs_n = fill(p, convert(Int64, (N-M)))
	probs_n = reshape(probs_n, length(probs_n), 1)
	if M%1 != 0 
		error("Number of processes should be a multiple of 2")
		MPI.Finalize()
	end
	N_m = 2^(N-M)
	l_n = convert(Int64, N_m)
	
	if id == 0 
		@printf("N: %d, np: %d\n", N, np)
		@printf("S: %f, sigma: %f, K: %f, r: %f, T: %f, %s\n", S, sigma, K, r, T, ifelse(opt_type==1, "Put", "Call"))
		tic()
	end
	
	v1 = 0.0
	#@printf("l_n: %d\n", l_n)
	#vt = zeros(Float64, l_n, 1)
	#p_vt = zeros(Float64, 1, l_n)
	p_vt = 0.0
	j = 0
	#for i in 0:(l_n-1)
	while j < nrep
	v1 = 0.0
	i = 0
	#srand(j)
	#smpl = sample(0:(l_n-1), n_sam, replace=true)
	while i < n_sam
		#node = smpl[i+1]
		#path = cat(2, integer_base_b(id, 2, M), integer_base_b(node, 2, N-M))
		path = cat(2, integer_base_b(id, 2, M), mapslices(simBer, probs_n, 2)')
		vt = calc_payoff(S, K, u, d, opt_type, path)
		v1 = v1 + vt
		i += 1
	end
	#v1 = exp(-r*T)*dot(vec(p_vt), vec(vt))
	#v1 = exp(-r*T)*v1*(1/n_sam)*calc_path_prob(integer_base_b(id, 2, M), probs_s)
	v1 = exp(-r*T)*v1*(1/n_sam)*(1/np)
	#@printf("id: %d, cnt: %d, v1: %f\n", id, j, v1)
	vals[j+1] = v1
	j += 1
	end
	MPI.Barrier(comm)
	
	#print(sprintf("P%d -- i=%d, v1: %f", id, i, v1))
	#@printf("Id: %d\n", id)
	#println(vals)
	reduced_v = MPI.Reduce(vals, MPI.SUM, 0, comm)
	if id == 0
		#@printf("Option Value: %f\n", reduced_v)
		#println(reduced_v)
		@printf("Mean of value: %f, Variance: %f\n", mean(reduced_v), var(reduced_v))
		@printf("Run time: %s\n", toc())
	end
	MPI.Finalize()
end

function calc_path_prob(path, probs) 
	p1 = prod(probs[find(path -> path==1, path)])
	p2 = prod(1-probs[find(path -> path==0, path)])
	return p1*p2
end

function calc_payoff(S, K, u, d, opt_type, path) 
	vt = 0.0
	if opt_type == 1 
		vt = K - S*prod(u[find(path -> path==1, path)])*prod(d[find(path -> path==0, path)])
	else 
		vt = S*prod(u[find(path -> path==1, path)])*prod(d[find(path -> path==0, path)]) - K
	end
	if vt < 0 
		vt = 0.0
	end
	return vt
end

function integer_base_b(x, b=2, ndigits = 0)
	#xi = as.integer(x)
	if(typeof(x) != Int64)
		error("ERROR: ", x,  " is not an integer")
	end
	N = length(x)
	xMax = x
	if ndigits == 0 & x > 0
		ndigits = convert(Int64, round(floor(log(2, xMax))+1))
	end
	ndigits = convert(Int64, round(ndigits))
	#@printf("ndigits: %d\n", ndigits)
	#@printf("N: %d\n", N)
	#Base.b = array(NA, dim=c(N, ndigits))
	Base_b = zeros(Int64, 1, ndigits)
	
	if ndigits == 0 
		return Base_b
	end
	for i in 1:ndigits
		Base_b[:,ndigits-i+1] = (x % b)
		#x = (x %/% b)
		x = div(x, b)
	end
	if N == 1
		return Base_b[1,: ]
	else 
	 return Base_b 
	end
end
