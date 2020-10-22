# Implementing Newton Raphson Method
#making list for eventual values of phi_d (note: phi_n is at index n-1)
list_phi = []


# Defining Function
def f(x, n):
    return x**(n+1) - x - 1
# Defining derivative of function
def g(x, n):
    return (n+1)*x**(n) - 1
    
def newtonraphson(x0,e,N, n):
    print('\n\n*** NEWTON RAPHSON METHOD IMPLEMENTATION ***')
    step = 1
    flag = 1
    condition = True
    while condition:
        if g(x0, n) == 0.0:
            print('Divide by zero error!')
            break
        
        x1 = x0 - f(x0, n)/g(x0, n)
        print('Iteration-%d, x1 = %0.20f and f(x1) = %0.20f' % (step, x1, f(x1, n)))
        x0 = x1
        step = step + 1
        
        if step > N:
            flag = 0
            break
        
        condition = abs(f(x1, n)) > e
    
    if flag==1:
        print('\nRequired root is: %0.20f' % x1)
    else:
        print('\nNot Convergent.')
    list_phi.append(x1)
#x0 = float(input('Enter Guess: '))
#e = float(input('Tolerable Error: '))
#N = int(input('Maximum Step: '))

# Starting Newton Raphson Method
#newtonraphson(x0,e,N)

def findphi_d(d):
    print('finding the solution to the equation x^(d+1) = x + 1 up to and including d' )
    n = 1
    while n < d + 1:
        print('n is: ', n)
        #Finding root for specific value of n
        newtonraphson(float(2), float(1/(10**14)), int(50), n)
        n = n + 1
    print(list_phi)
findphi_d(20)