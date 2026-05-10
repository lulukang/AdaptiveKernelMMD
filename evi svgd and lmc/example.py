import numpy as np
from environment import double_banana, sine, star_gaussian, banana, waveshape, crab,egaussian
import timeit
import time
from kernel import rbf_kernel
from stein_samplers import matrix_SVGD
import sklearn.metrics.pairwise as sk_metric
from mmd import square_mmd_fine
def trainer(model, method, n_particles, outer_iter):

    np.random.seed(0)
    d = model.dimension
    tau = 0.1
    #x = np.random.randn(n_particles, d)  # set initial particles as standard normal
    x = np.random.uniform(-1,1,[n_particles,d])
    x *= 4
    N = n_particles * d
    
    # comparision metric
    cross_entropy = []
    cross_entropy.append(-np.mean(env.logp(x)))



    if method == 'evi':
        # Evi
        x_initial = x
        dynamics = np.zeros([outer_iter+1, n_particles, d])
        dynamics[0,:,:]=x
        h = 0.1
        
        time_spent=[0]
        start=timeit.default_timer()
        for i in range(outer_iter):
            t = time.time()
            for j in range(100):

                Sqy = model.grad_log_p(x)
                diff = x[:, None, :] - x[None, :, :]
                kxy = np.exp(-np.sum(diff ** 2, axis=-1) / (2 * h ** 2)) / np.power(np.pi * 2.0 * h * h, d / 2)
                sumkxy = np.sum(kxy, axis=1, keepdims=True)
                gradK = -diff * kxy[:, :, None] / h ** 2  # N * N * 2
                dxkxy = np.sum(gradK, axis=0)  # N * 2
                obj = np.sum(np.transpose(gradK, axes=[1, 0, 2]) / sumkxy, axis=1)  # N * 2
                grad = (x - x_initial) / tau + (- dxkxy / sumkxy - obj - Sqy)
                grad_now = np.reshape(grad, (1, N))

                if np.sqrt(np.inner(grad_now, grad_now)) < 1e-9:
                    print(j)
                    break

                step_l = 1e-7
                # BB Step - length
                if j > 0:
                    y_k = grad_now - grad_old
                    s_k = np.reshape(x, (1, N)) - np.reshape(x_old, (1, N))
                    step_l = np.inner(s_k, s_k) / np.inner(s_k, y_k)

                grad_old = grad_now
                x_old = x
                x = x - step_l * grad

            elapsed = time.time() - t + time_spent[i]
            time_spent.append(elapsed)
            x_initial = x
            cross_entropy.append(-np.mean(env.logp(x)))
            dynamics[i+1, :, :] = x
                   
    elif method == "sgld":
        time_spent=[0]
        EVOLVEX = np.zeros([outer_iter+1, n_particles, d])
        a= 0.1
        b= 1
        c= 0.55
        EVOLVEX[0,:,:]=x
        # d = model.dimension
        # x = np.zeros([n, d])
        for i in range(outer_iter):

            t = time.time()
            epsilon = a * np.exp(np.log(b + i) * (- c))   # FDT
            x += epsilon * model.grad_log_p(x)
            x += np.sqrt(2 * epsilon) * np.random.randn(n_particles, d)

            cross_entropy.append(-np.mean(env.logp(x)))
            EVOLVEX[i+1, :, :] = x

            elapsed = time.time() - t + time_spent[i]

            print(elapsed)

            time_spent.append(elapsed)

            mmd.append(0)   # square_mmd_fine(p_samples, x, 'gaussian'))
            
            dynamics=EVOLVEX
            
    elif method == 'svgd':
        # SVGD
        time_spent=[0]
        start=timeit.default_timer()
        EVOLVEX = np.zeros([outer_iter+1, n_particles, d])
        adag = np.zeros([n_particles, d])  # for adagrad
        lr=0.1
        EVOLVEX[0,:,:]=x
        
        for i in range(outer_iter):
            print(i)
            t = time.time()
            ## Matrix Value
            '''
            grad_logp = model.grad_log_p(x)  # gradient information for each particles: n*d
            Hs = model.Hessian_log_p(x)  # Hessian information for each particles: ndd
            A = np.mean(Hs, axis=0)  # Average Hessian: d*d

            v = mixture_hessian_SVGD(x, grad_logp, Hs)
            '''


            grad_logp = model.grad_log_p(x)
            if i > 30:
                kernel = rbf_kernel(d, decay=True)
            else:
                kernel = rbf_kernel(d)
            B = np.eye(d)
            v = matrix_SVGD(x, grad_logp, kernel, B)



            adag += v ** 2  # update sum of gradient's square
            x = x + lr * v / np.sqrt(adag + 1e-12)



            cross_entropy.append(-np.mean(env.logp(x)))
            
            elapsed = time.time() - t + time_spent[i]
            time_spent.append(elapsed)
            EVOLVEX[i+1, :, :] = x
            
            #mmd.append(square_mmd_fine(p_samples, x, 'polynomial') + k_xi_xj)
        dynamics=EVOLVEX
        
    
    return dynamics, cross_entropy,time_spent  # cross_entropy


            # mmd.append(square_mmd_fine(p_samples, x, 'polynomial') + k_xi_xj)


if __name__ == '__main__':
    # Pick one toy environment to play
    #env_name = 'star'
    env_name='egaussian'
    #env_name='wave'


    if env_name == 'star':
        env = star_gaussian(100, 5)  # star gaussian mixture example
    elif env_name == 'wave':
        env = waveshape()
    elif env_name=='egaussian':
        env=egaussian()

    method = 'evi'
    n_particles = 200
    dim = 2
    repeat = 1
    #p_samples=env.sample(5000)
    #k_yi_yj = sk_metric.polynomial_kernel(p_samples, p_samples)
    #k_xi_xj=np.sum(k_yi_yj)/5000/5000
    outer_iter = 5000
    mmd=[]

    cross_entropy_results = np.zeros(shape=(repeat, outer_iter + 1))
    evolves1 = np.zeros(shape=(repeat, outer_iter+1, n_particles, dim))

    for rep in range(repeat):
        evolves1[rep, :, :, :], cross_entropy_results[rep, :] ,times1= trainer(env, method, n_particles, outer_iter)
        np.savez('evi_' + env_name, time=times1,evolves=evolves1[0, :, :, :], cross_entropy_results=np.mean(cross_entropy_results, axis=0))

    ngrid = 100
    # set the line space carefully to the region of your figure
    x = np.linspace(-3, 3, ngrid)
    y = np.linspace(-3, 3, ngrid)
    X, Y = np.meshgrid(x, y)
    Z = np.exp(env.logp(np.vstack((np.ndarray.flatten(X), np.ndarray.flatten(Y))).T)).reshape(ngrid, ngrid)
    np.savez(env_name, X=X, Y=Y, Z=Z)

    # method='svgd'
    # outer_iter=50000
    # cross_entropy_results = np.zeros(shape=(repeat, outer_iter + 1))
    # evolves2 = np.zeros(shape=(repeat, outer_iter+1, n_particles, dim))
    
    # for rep in range(repeat):
    #     evolves2[rep, :, :, :], cross_entropy_results[rep, :] ,times2= trainer(env, method, n_particles, outer_iter)
    #     np.savez('svgd_' + env_name, time=times2,evolves=evolves2[0, :, :, :], cross_entropy_results=np.mean(cross_entropy_results, axis=0))

    
    # method='sgld'
    # outer_iter=50000
    # cross_entropy_results = np.zeros(shape=(repeat, outer_iter + 1))
    # evolves3 = np.zeros(shape=(repeat, outer_iter+1, n_particles, dim))
    
    # for rep in range(repeat):
    #     evolves3[rep, :, :, :], cross_entropy_results[rep, :] ,times3= trainer(env, method, n_particles, outer_iter)
    #     np.savez('lmc_' + env_name, time=times3,evolves=evolves3[0, :, :, :], cross_entropy_results=np.mean(cross_entropy_results, axis=0))


