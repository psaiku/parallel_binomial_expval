import MPI

function cal_optvalue(N, S, sigma, K, r, T, opt_type)
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
	vt = 0.0
	p_vt = 0.0
	i = 0
	#for i in 0:(l_n-1)
	while i < l_n
		node = i
		#path = c(integer.base.b(id, b=2, ndigits=M), integer.base.b(node, b=2, ndigits=N-M))
		path = cat(2, integer_base_b(id, 2, M), integer_base_b(node, 2, N-M))
		#p_vt[i+1] = calc_path_prob(path, probs)
		#vt[i+1] = calc_payoff(S, K, u, d, opt_type, path)
		p_vt = calc_path_prob(path, probs)
		vt = calc_payoff(S, K, u, d, opt_type, path)
		v1 = v1 + p_vt*vt
		i += 1
	end
	#v1 = exp(-r*T)*dot(vec(p_vt), vec(vt))
	v1 = exp(-r*T)*v1
	#@printf("v1: %f\n", v1)
	MPI.Barrier(comm)
	
	#print(sprintf("P%d -- i=%d, v1: %f", id, i, v1))
	reduced_v = MPI.Reduce(v1, MPI.SUM, 0, comm)
	if id == 0
		@printf("Option Value: %f\n", reduced_v)
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
