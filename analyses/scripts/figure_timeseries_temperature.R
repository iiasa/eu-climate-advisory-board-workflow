# ******************************************************************************
# * Script for figure for temperature outcomes for Byers et al. (2023)
# * "Scenarios processing, vetting and feasibility assessment for the EU Scientific Advisory Board on Climate Change"
# *
# * Author: Jarmo S. Kikstra
# * Date last edited: May 30, 2023
# ******************************************************************************

meta.categories <- load_meta_eucab_clim_and_vet() %>%
  select(model, scenario, Category)

figure.rawdata <-
  meta.categories %>%
  left_join(
    load_var_eucab("AR6 climate diagnostics|Surface Temperature (GSAT)|MAGICCv7.5.3|50.0th Percentile") %>%
      bind_rows(
        load_var_eucab("AR6 climate diagnostics|Surface Temperature (GSAT)|MAGICCv7.5.3|5.0th Percentile")
      ) %>%
      bind_rows(
        load_var_eucab("AR6 climate diagnostics|Surface Temperature (GSAT)|MAGICCv7.5.3|95.0th Percentile")
      )
  )


figure.data.scenariorange <- figure.rawdata %>%
  filter(variable == "AR6 climate diagnostics|Surface Temperature (GSAT)|MAGICCv7.5.3|50.0th Percentile") %>%
  group_by(Category, year, variable) %>%
  summarise(
    median = median(value),
    p5 = quantile(value, probs = 0.05),
    p95 = quantile(value, probs = 0.95),
  )

figure.data.combinedrange <- figure.rawdata %>%
  group_by(Category, year) %>%
  summarise(
    median = median(value),
    p5 = quantile(value, probs = 0.05),
    p95 = quantile(value, probs = 0.95),
  )


p.scenariorange <- ggplot(
  figure.data.scenariorange,
  aes(x = year, fill = Category)
) +
  # facet_grid(~Category) +
  geom_line(
    aes(
      y = median
    ),
    alpha = 0.2
  ) +
  geom_ribbon(
    aes(
      y = median,
      ymin = p5,
      ymax = p95
    ),
    alpha = 0.2
  ) +
  geom_text(y = 1.55, x = 2015, data = data.frame(Category = "C1"), aes(Category = "C1"), label = "1.5\u00b0C", vjust = 0) +
  geom_hline(yintercept = 1.5, linetype = "dotted") +
  theme_classic() +
  scale_fill_manual(
    values = c(
      get_ipcc_colours("c.hex"),
      "lightgrey"
    ),
    breaks = c(get_ipcc_colours("c.list"), "Full database")
  ) +
  scale_colour_manual(
    values = c(
      get_ipcc_colours("c.hex"),
      "lightgrey"
    ),
    breaks = c(get_ipcc_colours("c.list"), "Full database")
  ) +
  xlab(NULL) +
  ylab("Temperature above pre-industrial [\u00b0C]") +
  scale_x_continuous(breaks = c(2020, 2050, 2100)) +
  theme_hc() +
  labs(
    subtitle = "Scenario range",
    caption = "5th-95th percentile range within each category. Median highlighted."
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

p.scenariorange


p.combinedrange <- ggplot(
  figure.data.combinedrange,
  aes(x = year, fill = Category)
) +
  facet_grid(~Category) +
  geom_line(
    aes(
      y = median
    ),
    alpha = 0.2
  ) +
  geom_ribbon(
    aes(
      y = median,
      ymin = p5,
      ymax = p95
    ),
    alpha = 0.2
  ) +
  geom_text(y = 1.55, x = 2015, data = data.frame(Category = "C1"), aes(Category = "C1"), label = "1.5\u00b0C", vjust = 0) +
  geom_hline(yintercept = 1.5, linetype = "dotted") +
  theme_classic() +
  scale_fill_manual(
    values = c(
      get_ipcc_colours("c.hex"),
      "lightgrey"
    ),
    breaks = c(get_ipcc_colours("c.list"), "Full database")
  ) +
  scale_colour_manual(
    values = c(
      get_ipcc_colours("c.hex"),
      "lightgrey"
    ),
    breaks = c(get_ipcc_colours("c.list"), "Full database")
  ) +
  xlab(NULL) +
  ylab("Temperature above pre-industrial [\u00b0C]") +
  scale_x_continuous(breaks = c(2020, 2050, 2100)) +
  theme_hc() +
  labs(subtitle = "Combined range (climate uncertainty and scenario range)") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

p.combinedrange

p <- ((p.scenariorange + theme(legend.position = "none")) / p.combinedrange) +
  plot_layout(heights = c(3, 1)) +
  plot_annotation(tag_levels = "A")

p

ggsave(
  plot = p,
  filename = here("analyses", "figure_timeseries_temperature.png"),
  width = 200,
  height = 300,
  unit = "mm",
  dpi = 500
)
