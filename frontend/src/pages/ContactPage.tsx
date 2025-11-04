import PageSection from '../components/PageSection';

const contactOptions = [
  {
    label: 'Email',
    value: 'nik52.lab@gmail.com',
    hint: 'Share context, datasets, timelines and current tooling.'
  },
  {
    label: 'Telegram',
    value: '@nik52_lab',
    hint: 'Quick sync for prototypes, model audits or discovery calls.'
  },
  {
    label: 'GitHub',
    value: 'github.com/nik52',
    hint: 'Open source explorations, issue tracking and changelogs.'
  }
];

const ContactPage = () => {
  return (
    <div className="page-stack">
      <PageSection
        subtitle="Contact"
        title="Work with Nik52"
        description="Reach out with a short brief or choose a live sync channel. I respond within 24 hours on weekdays."
      >
        <div className="contact-grid">
          {contactOptions.map((option) => (
            <div className="contact-card" key={option.label}>
              <span className="badge">{option.label}</span>
              <strong>{option.value}</strong>
              <span>{option.hint}</span>
            </div>
          ))}
        </div>
      </PageSection>

      <PageSection
        subtitle="Expectations"
        title="How collaborations usually start"
        description="We align on goals, set measurable outcomes and design a lean launch roadmap."
      >
        <div className="timeline">
          <div className="timeline__item">
            <h3 className="timeline__title">Discovery workshop</h3>
            <p className="timeline__meta">Week 1</p>
            <p className="card__description">
              Clarify target metrics, constraints, available data and business drivers behind the project.
            </p>
          </div>
          <div className="timeline__item">
            <h3 className="timeline__title">Solution blueprint</h3>
            <p className="timeline__meta">Week 2</p>
            <p className="card__description">
              Produce an architecture map, experimentation backlog and risk matrix tailored to your stack.
            </p>
          </div>
          <div className="timeline__item">
            <h3 className="timeline__title">Pilot delivery</h3>
            <p className="timeline__meta">Weeks 3-6</p>
            <p className="card__description">
              Launch a measurable MVP, collect telemetry and iterate on top priorities before widescale rollout.
            </p>
          </div>
        </div>
      </PageSection>
    </div>
  );
};

export default ContactPage;
