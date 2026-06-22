from typing import TypedDict
from langchain.tools import tool
import numexpr
import os
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
import sympy
import math
import numpy as np
from sympy.abc import _clash




local_env = {**_clash, **sympy.__dict__}




#general tool for fast calc using numexpr
@tool
def calculator(expression:str)->str:
    """
    Evaluate a SINGLE numexpr mathematical expression.

    INPUT MUST BE ONLY AN EXPRESSION.

    Valid:
    2+2
    sqrt(16)
    sin(pi/2)
    sin(90*pi/180)

    Invalid:
    import math
    math.sin(90)
    result = 2+2
    print(2+2)
    ```
    """

    local_dict = {
    "pi": math.pi,
    "e": math.e,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "sqrt": np.sqrt,
    "log": np.log,
}

    try:
        return str(
            numexpr.evaluate(
                    expression.strip(),
                            global_dict={},  
                            local_dict=local_dict,  # add common mathematical functions
                        )
                )
    except Exception as e:
        return str(e)


#calculus tools

@tool
def differentiate_expression(expression:str,order_of_differention:list[str])->str:
    """
    returns indefinitive derivative
    given a corrrect sympy format expression and list of variables(order_of_differention) by which we have to differentiate 

    Rules:
    1.expression which we have to differentiate without differential operator must be in sympy library format 
    example for d(2x^2-2x+2)/dx or d(2xsquare-2x+two or related)dx -   2*x**2-2*x+2

    2.second parameter order_of_differention should contains variables by which we have to differentiate in exact order of differentation
    example-  for d3(2^xyz)/dz2dy3dx - [z,z,y,y,y,x]


    """
    try:
        transformation=standard_transformations+(implicit_multiplication_application,)

        expr=parse_expr(expression,transformations=transformation,global_dict=local_env)
        vars_to_diff = []

        for v in order_of_differention:
            vars_to_diff.append(sympy.Symbol(v))
   

        return sympy.diff(expr,*vars_to_diff)

    except Exception as e:
        return f"Error : {e}"


from typing_extensions import TypedDict,NotRequired

class IntegralSpec(TypedDict):
    """
    Indefinite:
    {"variable":"x"}

    Definite:
    {"variable":"x","lower":"0","upper":"1"}
    """
    variable: str
    lower: NotRequired[str]
    upper: NotRequired[str]


@tool  
def integration_nonnumeric(expression:str,list_of_integral:list[IntegralSpec])->str:
    """
    Integrates a SymPy expression.
    

    Parameters
    ----------
    expression:
        Expression in SymPy format.
        example for inegral (2x^2-2x+2)dx or 2xsquare-2x+two or related dx -   2*x**2-2*x+2

    integrals:
        Ordered list of integration specifications.

        Indefinite integral:
        [{"variable":"x"}]

        Definite integral:
        [{"variable":"x","lower":"-1","upper":"1"}]

        Multiple integrals:
        [
            {"variable":"x","lower":"-oo","upper":"oo"},
            {"variable":"y","lower":"-oo","upper":"oo"}
        ]
        Integration is performed in the order provided.
    
    """

    try:
        transformation=standard_transformations+(implicit_multiplication_application,)
        expr=parse_expr(expression,transformations=transformation,global_dict=local_env)

        print(type(list_of_integral))
        print(repr(list_of_integral))

        integration_args=[]
        for item in list_of_integral:
            var=sympy.Symbol(item['variable'])

            if 'lower' in item and 'upper' in item:

                lower = parse_expr(
                    item["lower"],
                    transformations=transformation,
                    global_dict=local_env
                )

                upper = parse_expr(
                    item["upper"],
                    transformations=transformation,
                    global_dict=local_env
                )

                integration_args.append(
                    (var, lower, upper)
                )

            else:
                integration_args.append(var)
            
        result=sympy.integrate(
            expr,*integration_args
            )

        return str(result)

    except Exception as e:
        return f"Error : {e}"


@tool
def solve_limits(func:str,x:str,x_nod:str,directon:str=None)->str:
    """for given function (example f(x)) , x and x_nod(tending value) this function returns value of limit x tending to x_nod of function f(x).

    Rules:
     1.expression/function for which we have to find limiting value must be in sympy library format 
    example for (2x^2-2x+2) or (2xsquare-2x+two or related) -   2*x**2-2*x+2
    2. to evaluate a limit at one side only, pass '+' or '-' as a fourth argument called direction. For example, to compute
        lim x tending to 0+ for f(x)=1/x - function=1/x,x=x,x_nod=0,direction='+'
    """


    try:
        transformation=standard_transformations+(implicit_multiplication_application,)
        expr=parse_expr(func,transformations=transformation,global_dict=local_env)

        if directon is None:
            return sympy.limit(expr,x,x_nod)
        else:
            return sympy.limit(expr,sympy.Symbol(x),float(x_nod),str(directon))

    except Exception as e:
        return f"error {e}"


@tool
def find_series_expansion(expression:str,x_nod:str,n:str)->str:
    """for given valid expression/function,x_nod,n this function returns series expansion of given expression around the point x=x_nod terms of order x^n.
        Rules:
     1.expression/function for which we have to find limiting value must be in sympy library format 
    example for (2x^2-2x+2) or (2xsquare-2x+two or related) -   2*x**2-2*x+2
    2. example to find expansion of e^sinx around the point x=x_nod terms of order x^4  - x_nod=0 and n=4 
    3. last point n is ommited i.e all terms of degree n and higher are omitted.¯
    4.if asked about find expansion including nth term or till nth term pass n=n+1 because nth term is ommited.
    
    
    """


    try:
        transformation=standard_transformations+(implicit_multiplication_application,)
        expr=parse_expr(expression,transformations=transformation,global_dict=local_env)



        return expr.series(sympy.Symbol('x'),float(x_nod),float(n))

    except Exception as e:
        return f"error {e}"







@tool
def solve_one_variable_equations(sympy_equation:str)->str:
    """solves one variable equations and gives satisfying values in a List
     Requirement:
    1.equation must be in sympy library format
     example for 2x^2-2x+2 or 2xsquare-2x+two or related -   2*x**2-2*x+2
        """
    transformation=standard_transformations+(implicit_multiplication_application,)
    expr = None
    variables = set()
    try:
        if '=' in sympy_equation:
            lhs,rhs=sympy_equation.split('=',1)
            lhs=parse_expr(lhs.strip(),transformations=transformation,global_dict=local_env)
            rhs=parse_expr(rhs.strip(),transformations=transformation,global_dict=local_env)
            expr=sympy.Eq(lhs,rhs)
        else:
            expr=parse_expr(sympy_equation,transformations=transformation,global_dict=local_env)
        variables=expr.free_symbols
        solutions=sympy.solve(expr,list(variables),dict=True)
        if len(solutions)>0:
            return str(solutions)
    except Exception as e:
        if expr is None:
            return f"Error: could not parse equation: {e}"
    # Numerical fallback
    if expr is None:
        return "Error: could not parse the equation."
    if not variables:
        return "Error: no variables found in the equation."

    var = list(variables)[0]

    # Wider range of guesses 
    guesses = [0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 5.0, -5.0, 10.0, -10.0]

    for guess in guesses:
        try:
       
            num_sol = sympy.nsolve(expr, var, guess)
            formatted_sol = {str(var): float(num_sol)}
            return f"Numerical solution: {formatted_sol}"
        except Exception:
            continue

    return "Could not find any real solutions symbolically or numerically."



@tool
def solve_multi_variable_equations(sympy_equation:str)->str:
    """solves more than one variable equations and gives satisfying values
     Requirement:
     1.system of equations must be provided in a single string nothing else and each equation must be seprated by comma
        example - for a 3 variable problem 3 equations must be - "2*x-y+z-3,x+y+z-6,x+2*y-3*z+4"
     2.equation must be in sympy library format
      example - for 2x^2-2x+2 or 2xsquare-2x+two or related -   2*x**2-2*x+2
    3. Equation passed must be a string not a dict key or value not a list just a single string seprated by commas
        """
    transformation=standard_transformations+(implicit_multiplication_application,)
 
    parsed_expressions=[]
    variables_in_eq=set()
    try:
        raw_equations=[eq.strip() for eq in sympy_equation.split(',')]
        for eq in raw_equations:
            if '=' in eq:
                lhs,rhs=eq.split('=',1)
                lhs=parse_expr(lhs.strip(),transformations=transformation,global_dict=local_env)
                rhs=parse_expr(rhs.strip(),transformations=transformation,global_dict=local_env)
                expr=sympy.Eq(lhs,rhs)
            else:
                expr=parse_expr(eq,transformations=transformation,global_dict=local_env)
            parsed_expressions.append(expr)
            variables_in_eq.update(expr.free_symbols)
        if not parsed_expressions:
            return "Error: no valid equations extracted from given string."
        solutions=sympy.solve(parsed_expressions,list(variables_in_eq),dict=True)
        if len(solutions)>0:
            return str(solutions)
    except Exception as e:
        if not parsed_expressions or not variables_in_eq:
            return f"Error: could not parse equations: {e}"
    # Numerical fallback
    if not parsed_expressions or not variables_in_eq:
        return "Error: no valid equations or variables found."
    num_vars = len(variables_in_eq)

    guesses = [
        [0.0] * num_vars, [1.0] * num_vars, [-1.0] * num_vars,
        [0.5] * num_vars, [-0.5] * num_vars,
        [2.0] * num_vars, [-2.0] * num_vars,
        [5.0] * num_vars, [-5.0] * num_vars,
        [10.0] * num_vars, [-10.0] * num_vars,
    ]
    for guess in guesses:
        try:
            num_sol = sympy.nsolve(parsed_expressions, list(variables_in_eq), guess)
            formatted_sol = {str(var): float(val) for var, val in zip(list(variables_in_eq), num_sol)}
            return f"Numerical solution: {formatted_sol}"
        except Exception:
            continue
    return "Could not find any real solutions symbolically or numerically."


