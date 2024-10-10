import torch
import pandas as pd
from torch.autograd.functional import jacobian

def first_expression(p, Tp, Lambda, h_0):
    P1 = 1 - torch.exp(-Lambda * Tp)
    A = p * (1 - P1) / (1 - p * P1)
    return (1 - A) * h_0

def second_expression(p, Tp, Tnp, Lambda, W1, W2, h_0):
    P1 = 1 - torch.exp(-Lambda * Tp)
    P2 = 1 - torch.exp(- Lambda * Tnp)

    E_AT = 1 / Lambda

    C = (1 - P2) + (2 * p * P1 * (1 - P1)) / (1 - p * P1)

    Gamma_0 = (p * P1) ** 2 * W1 * (Tnp - E_AT) / (1 - p * P1)

    # Γ1
    Gamma_1 = W1 * (1 - P1) * (
        (2 * p * P2 * (Tnp - E_AT)) / (1 - 2 * p * P2)
        + ((2 * p * P2) ** 2 * (Tp - E_AT)) / (1 - 2 * p * P2) ** 2
    ) + W1 * P2 * (1 - p) * (
        (Tnp - E_AT) * p * P1 / (1 - p * P1)
        + (Tp - E_AT) * ((p * P1) ** 2 / (1 - p * P1) ** 2)
    )

    # K1
    K1 = W2 * (E_AT + Tp * p * P1 / (1 - p * P1))

    # K2
    K2 = W2 * (Tp - E_AT) / (1 - p * P1)

    # K3
    K3 = W2 * (
        (Tnp * ((1 - P2) + (2 * p * (1 - P1) * P2) / (1 - 2 * p * P2)))
        + Tp * ((2 * p * P1 * (1 - P1)) / (1 - 2 * p * P2) + (2 * p * (1 - p) * P1 ** 2) / (1 - p * P1) ** 2)
    )

    # K4
    K4 = W2 * (Tnp - E_AT) * P2 * (1 - p) / (1 - p * P1)

    # δ
    delta = Gamma_1 - Gamma_0 + K3 - K4 - K1
    
    return (Gamma_1 - Gamma_0 + K3 - K4 - K1 - C * h_0)


def get_derivatives(p, Tp, Tnp, Lambda, W1, W2, h_0):
    
    p = torch.tensor(p, dtype=torch.float32, requires_grad=True)
    Tp = torch.tensor(Tp, dtype=torch.float32, requires_grad=True)
    Tnp = torch.tensor(Tnp, dtype=torch.float32, requires_grad=True)
    Lambda = torch.tensor(Lambda, dtype=torch.float32, requires_grad=True)
    W1 = torch.tensor(W1,dtype=torch.float32, requires_grad=True)
    W2 = torch.tensor(W2, dtype=torch.float32, requires_grad=True)
    h_0 = torch.tensor(h_0, dtype=torch.float32, requires_grad=True)

    first_expression_derivatives = jacobian(first_expression, (p,Tp, Lambda, h_0))

    second_expression_derivatives = jacobian(second_expression, (p, Tp, Tnp, Lambda, W1, W2, h_0))

    return first_expression_derivatives, second_expression_derivatives

parameter_map = {
    "first_expression" : {"p" : 0,
                "Tp" : 1,
                "Lambda" :2,
                "h_0" : 3},
    "second_expression" : {
        "p" : 0,
        "Tp" : 1,
        "Tnp" : 2,
        "Lambda" : 3,
        "W1" : 4,
        "W2" : 5,
        "h_0" : 6
    }
}

def process_derivatives(derivative : torch.tensor):
    if derivative.ndim == 2 and derivative.shape[0] == derivative.shape[1]:
        return torch.diag(derivative)
    return derivative

if __name__ == "__main__":
    derivative_parameter = "Lambda"
    parameters = {
        "p" : 0.4,
        "Tp" : 4,
        "Tnp" : 6,
        "Lambda" : [0.2, 0.4, 0.7],
        "W1" : 3,
        "W2" : 2,
        "h_0" : 1
    }
        
    first_expression_derivatives, second_expression_derivatives = get_derivatives(**parameters)        

    first_derivatives = first_expression_derivatives[parameter_map['first_expression'][derivative_parameter]]
    first_derivatives = process_derivatives(first_derivatives)

    second_derivatives = second_expression_derivatives[parameter_map['second_expression'][derivative_parameter]]
    second_derivatives = process_derivatives(second_derivatives)

    rows = []

    for ind, D in enumerate(list(zip(first_derivatives, second_derivatives))):
        fd,sc = D
        data = {"Derivative_Quantity_1" : fd.item(), "Derivative_Quantity_2" : sc.item(), "derivative_parameter" : derivative_parameter}
        for parameter,value in parameters.items():
            if isinstance(value, list):
                data[parameter] = value[ind]
            else:
                data[parameter] = value
        rows.append(data)    

    result = pd.DataFrame(rows, columns = ["p", "Tp", "Tnp", "Lambda", "W1","W2", "h_0" , "Derivative_Quantity_1", "Derivative_Quantity_2", "derivative_parameter"])
    result.to_csv("Derivatives.csv", index=False)
