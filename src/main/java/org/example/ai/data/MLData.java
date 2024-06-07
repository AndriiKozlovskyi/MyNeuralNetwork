package org.example.ai.data;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
@AllArgsConstructor
public class MLData {
    private double[] inputs;
    private double[] targets;
}
